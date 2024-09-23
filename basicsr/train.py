import argparse
import datetime
import logging
import math
import random
import time
import warnings
from os import path as osp

import torch

from basicsr.data import (CPUPrefetcher, CUDAPrefetcher, EnlargedSampler,
                          build_dataloader, build_dataset)
from basicsr.models import build_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, init_tb_logger,
                           init_wandb_logger, make_exp_dirs,
                           mkdir_and_rename, parse_options, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist

# Suppress specific UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)


def parse_args_and_options(root_path, is_train=True):
    parser = argparse.ArgumentParser(description='Training script options.')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='Job launcher type.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training.')
    args = parser.parse_args()

    # Parse options from the YAML file
    opt = parse_options(args.opt, root_path, is_train=is_train)

    # Distributed training settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Distributed training is disabled.', flush=True)
    else:
        opt['dist'] = True
        dist_params = opt.get('dist_params', {})
        init_dist(args.launcher, **dist_params)

    opt['rank'], opt['world_size'] = get_dist_info()

    # Set random seed for reproducibility
    seed = opt.get('manual_seed') or random.randint(1, 10000)
    opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def initialize_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger('basicsr', logging.INFO, log_file)
    logger.info(get_env_info())
    logger.info(parse_options.dict2str(opt))

    # Initialize WandB and TensorBoard loggers
    if opt['logger'].get('wandb') and opt['logger']['wandb'].get('project'):
        if not opt['logger'].get('use_tb_logger'):
            raise ValueError('TensorBoard logger must be enabled when using WandB.')
        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger'):
        tb_logger = init_tb_logger(osp.join('tb_logger', opt['name']))

    return logger, tb_logger


def create_dataloaders(opt, logger):
    train_loader, val_loader = None, None
    total_epochs, total_iters = 0, 0

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset = build_dataset(dataset_opt)
            sampler = EnlargedSampler(
                dataset, opt['world_size'], opt['rank'], dataset_opt.get('dataset_enlarge_ratio', 1))
            train_loader = build_dataloader(
                dataset, dataset_opt, opt['num_gpu'], opt['dist'], sampler, opt['manual_seed'])
            num_iter_per_epoch = math.ceil(
                len(dataset) * dataset_opt.get('dataset_enlarge_ratio', 1) /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = opt['train']['total_iter']
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)

            logger.info(
                f"Training statistics:\n"
                f"\tNumber of train images: {len(dataset)}\n"
                f"\tDataset enlarge ratio: {dataset_opt.get('dataset_enlarge_ratio', 1)}\n"
                f"\tBatch size per GPU: {dataset_opt['batch_size_per_gpu']}\n"
                f"\tWorld size (number of GPUs): {opt['world_size']}\n"
                f"\tIterations per epoch: {num_iter_per_epoch}\n"
                f"\tTotal epochs: {total_epochs}; Total iterations: {total_iters}.")

        elif phase == 'val':
            dataset = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                dataset, dataset_opt, opt['num_gpu'], opt['dist'], None, opt['manual_seed'])
            logger.info(f"Number of validation images/folders in {dataset_opt['name']}: {len(dataset)}")
        else:
            raise ValueError(f"Unknown dataset phase: {phase}")

    return train_loader, val_loader, total_epochs, total_iters


def main():
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt = parse_args_and_options(root_path, is_train=True)

    torch.backends.cudnn.benchmark = True

    # Load resume state if available
    resume_state = None
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))

    # Create necessary directories
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # Initialize loggers
    logger, tb_logger = initialize_loggers(opt)

    # Create dataloaders
    train_loader, val_loader, total_epochs, total_iters = create_dataloaders(opt, logger)
    train_sampler = train_loader.sampler if hasattr(train_loader, 'sampler') else None

    # Build model and resume training if applicable
    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = build_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch {resume_state['epoch']}, iteration {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = build_model(opt)
        start_epoch, current_iter = 0, 0

    # Initialize message logger
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # Set up prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
    if prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        if not opt['datasets']['train'].get('pin_memory', False):
            raise ValueError('pin_memory must be True when using CUDAPrefetcher.')
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info('Using CUDA prefetcher for data loading.')
    else:
        raise ValueError(f"Invalid prefetch_mode: {prefetch_mode}")

    logger.info(f"Starting training from epoch {start_epoch}, iteration {current_iter + 1}")

    # Training loop
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time_start = time.time()
            current_iter += 1
            if current_iter > total_iters:
                break

            # Update learning rate
            model.update_learning_rate(current_iter, opt['train'].get('warmup_iter', -1))

            # Optimize parameters
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            # Logging
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {
                    'epoch': epoch,
                    'iter': current_iter,
                    'lrs': model.get_current_learning_rate(),
                    'time': time.time() - data_time_start,
                    'data_time': data_time_start - start_time,
                    **model.get_current_log(),
                }
                msg_logger(log_vars)

            # Save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # Validation
            if (opt.get('val') and opt['datasets'].get('val') and
                    current_iter % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger, opt['val'].get('save_img', False))

            start_time = time.time()
            train_data = prefetcher.next()

    # End of training
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training completed. Total time: {total_time}")
    logger.info('Saving the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if opt.get('val') and opt['datasets'].get('val'):
        model.validation(val_loader, current_iter, tb_logger, opt['val'].get('save_img', False))
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
