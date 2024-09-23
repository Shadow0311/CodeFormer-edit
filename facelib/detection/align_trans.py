import cv2
import numpy as np

# Reference facial points, a list of coordinates (x, y)
REFERENCE_FACIAL_POINTS = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}: {}'.format(__file__, super().__str__())


def get_reference_facial_points(output_size=None, inner_padding_factor=0.0, outer_padding=(0, 0), default_square=False):
    """
    Function:
    ----------
        Get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square:
                crop_size = (112, 112)
            else:
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor on each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            Size of aligned face image
        @inner_padding_factor: float
            Padding factor for inner padding (0 <= factor <= 1)
        @outer_padding: (w_pad, h_pad)
            Padding outside the cropped area
        @default_square: bool
            If True:
                default crop_size = (112, 112)
            Else:
                default crop_size = (96, 112)
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            Each row is a pair of transformed coordinates (x, y)
    """
    tmp_5pts = REFERENCE_FACIAL_POINTS.copy()
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE, dtype=np.float32)

    # 0) Make the inner region a square
    if default_square:
        size_diff = np.max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if output_size and (output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if inner_padding_factor == 0 and outer_padding == (0, 0):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException('No paddings to apply, output_size must be None or {}'.format(tmp_crop_size))

    # Check inner padding factor
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('inner_padding_factor must be between 0 and 1.0')

    # Calculate output_size if not provided
    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0) and output_size is None):
        output_size = (tmp_crop_size * (1 + inner_padding_factor * 2)).astype(np.int32)
        output_size += np.array(outer_padding)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException('outer_padding must be smaller than output_size')

    # 1) Pad the inner region according to inner_padding_factor
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # 2) Resize the padded inner region
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Aspect ratio mismatch after applying padding')

    scale_factor = size_bf_outer_pad[0] / tmp_crop_size[0]
    tmp_5pts *= scale_factor
    tmp_crop_size = size_bf_outer_pad

    # 3) Add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)

    return reference_5point


def get_similarity_transform_for_cv2(src_pts, dst_pts):
    """
    Compute similarity transform from src_pts to dst_pts.

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            Source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            Destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            Transformation matrix from src_pts to dst_pts
    """
    tfm, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if tfm is None:
        raise FaceWarpException("Could not estimate similarity transform")
    return tfm


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Compute affine transform matrix 'tfm' from src_pts to dst_pts.

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            Source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            Destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            Transformation matrix from src_pts to dst_pts
    """
    tfm, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
    if tfm is None:
        raise FaceWarpException("Could not estimate affine transform")
    return tfm


def warp_and_crop_face(src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type='similarity'):
    """
    Apply affine transform to the face image based on facial landmarks.

    Parameters:
    ----------
        @src_img: np.array
            Input image
        @facial_pts: list or np.array
            Detected facial landmarks in the source image
        @reference_pts: list or np.array
            Reference facial landmarks
        @crop_size: (w, h)
            Output face image size
        @align_type: str
            Transform type: 'similarity', 'cv2_affine', or 'affine'
    Returns:
    ----------
        @face_img: np.array
            Output face image with size (w, h) = crop_size
    """
    if reference_pts is None:
        # Use default reference points
        reference_pts = get_reference_facial_points(output_size=crop_size)

    ref_pts = np.float32(reference_pts)
    src_pts = np.float32(facial_pts)

    if ref_pts.shape != src_pts.shape:
        raise FaceWarpException('facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:  # Default to 'similarity'
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)

    if tfm is None:
        raise FaceWarpException('Transformation matrix is None')

    face_img = cv2.warpAffine(src_img, tfm, crop_size)

    return face_img
