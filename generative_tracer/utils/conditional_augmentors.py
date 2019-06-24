# Standard library imports
import itertools

# Third party imports
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, restoration, feature, transform

# Local application imports


# TODO: Scikit-Image has limited ability to apply filters to 3D images,
# thus I am using scipy.ndimage in a few places. Reference this issue
# in skimage for determining whether the filters needed below are
# available for 3D images:
# https://github.com/scikit-image/scikit-image/issues/2247

# Work for arbitrary input dimensions:


def ball_shape(image, radius):
    """Computes an d-dimensional ball for use in other filters where d
    is given by the shape of the image.

    Keyword arguments:
        image -- an ndarray representing the image for which the mask
        is being generated.
        radius -- a floatint point value specifying how large the ball
        should be.

    Returns:
        ndarray: a mask to use as a template surrounding a given pixel
        in other filters.
    """
    d = len(image.shape)
    mask = None
    if d == 2:
        mask = morphology.disk(radius)
    elif d == 3:
        mask = morphology.ball(radius)
    return mask


def mean_filter(image, radius=1):
    """Computes the mean around each pixel of an image by looking at
    a disk of a user-specified radius surrounding it.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied
        radius -- size of area to consider for each pixel

    Returns:
        ndarray: an image that is the same size as the original image
        representing the mean at each pixel in the original image.
    """
    mask = ball_shape(image, radius)
    return ndimage.generic_filter(image, np.mean, footprint=mask)


def variance_filter(image, radius=1):
    """Computes the variance around each pixel of an image by looking at
    a disk of a user-specified radius surrounding it.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied
        radius -- size of area to consider for each pixel

    Returns:
        ndarray: an image that is the same size as the original image
        representing the variance at each pixel in the original image.
    """
    return np.power(image - mean_filter(image, radius), 2)


def median_filter(image, radius=1):
    """Computes the median around each pixel of an image by looking at
    a disk of a user-specified radius surrounding it.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied
        radius -- size of area to consider for each pixel

    Returns:
        ndarray: an image that is the same size as the original image
        representing the median at each pixel in the original image.
    """
    mask = ball_shape(image, radius)
    return ndimage.median_filter(image, footprint=mask)


def minimum_filter(image, radius=1):
    """Computes the minimum around each pixel of an image by looking at
    a disk of a user-specified radius surrounding it.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied
        radius -- size of area to consider for each pixel

    Returns:
        ndarray: an image that is the same size as the original image
        representing the minimum at each pixel in the original image.
    """
    mask = ball_shape(image, radius)
    return ndimage.minimum_filter(image, footprint=mask)


def maximum_filter(image, radius=1):
    """Computes the maximum around each pixel of an image by looking at
    a disk of a user-specified radius surrounding it.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied
        radius -- size of area to consider for each pixel

    Returns:
        ndarray: an image that is the same size as the original image
        representing the maximum at each pixel in the original image.
    """
    mask = ball_shape(image, radius)
    return ndimage.maximum_filter(image, footprint=mask)


def gaussian_blur_filter(image, sigma=2):
    """Computes a gaussian blur over the original image with a scale
    parameter specified by sigma.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied
        sigma -- shape parameter specifying the area of influence for
        each pixel

    Returns:
        ndarray: an image that is the same size as the original image
        representing a blurred version of the original image.
    """
    return filters.gaussian(image, sigma=sigma, multichannel=True)


def difference_of_gaussians_filter(image, sigma1, sigma2):
    """Computes a difference of two gaussian blurred images over the
    original image. Two scale parameters give the shapes of each
    Gaussian. For a refresher on math terminology used for the keyword
    arguments:

    minuend − subtrahend = difference

    Keyword arguments:
        image -- an ndarray representing the image to which the filter
        will be applied
        sigma1 -- shape parameter specifying the area of influence for
        each pixel for the minuend blurred image
        sigma2 -- shape parameter specifying the area of influence for
        each pixel for the subtrahend blurred image

    Returns:
        ndarray: an image that is the same size as the original image
        representing a blurred version of the original image.
    """
    blur1 = gaussian_blur_filter(image, sigma=sigma1)
    blur2 = gaussian_blur_filter(image, sigma=sigma2)
    return blur1 - blur2


def laplacian_filter(image):
    """Computes the Laplace operator over the original image.

    Keyword arguments:
        image -- an ndarray representing the image to which the filter
        will be applied

    Returns:
        ndarray: an image that is the same size as the original image
        representing the Laplacian of the original image.
    """
    return filters.laplace(image, ksize=3)


def neighbor_filter(image, max_shift=3):
    """Create a list of filters by shifting the image a single pixel at
    a time in every direction up to a maximum shift of max_shift. This
    includes all combination of diagonal shifts.

    Keyword arguments:
        image -- an ndarray representing the image to which the filter
        will be applied
        max_shift -- The maximum number of pixels the image will be
        shifted

    Returns:
        list(ndarray): a list of images that are the same size as the
        original image but shifted in different directions.
    """
    neighbor_images = []
    d = len(image.shape)
    directions = list(itertools.product([-1, 0, 1], repeat=d))
    for t in directions:
        if np.sum(np.abs(t)) == 0:
            continue
        for sigma in range(1, max_shift):
            shift = tuple([sigma * val for val in t])
            neighbor_images.append(ndimage.shift(image, shift))
            # tform = transform.SimilarityTransform(
            #     scale=1,
            #     rotation=0,
            #     translation=tuple([sigma * val for val in t]),
            # )
            # neighbor_images.append(transform.warp(image, tform))
    return neighbor_images

def sobel_filter(image):
    """Create the Sobel filter over an input image

    Keyword arguments:
        image -- an ndarray representing the image to which the filter
        will be applied

    Returns:
        ndarray: an image that has the same size as the
        original image but the contents of the pixels are the result of
        the Sobel filter.
    """
    # For 3D compatibility of skimage, see here:
    # https://github.com/scikit-image/scikit-image/pull/2787
    # return filters.sobel(image)
    return ndimage.sobel(image)


# Specific to 2-dimensional images:


def membrane_projection_2d_filter(
    image,
    membrane_kernel_size,
    operators=[np.sum, np.mean, np.std, np.median, np.max, np.min],
):
    """ TODO
    """
    membrane_kernel = np.zeros((membrane_kernel_size, membrane_kernel_size))
    membrane_kernel[:, membrane_kernel_size // 2] = 1.
    test_kernels = []
    for i, angle in enumerate(range(0, 180, 6)):
        test_kernels.append(
            transform.rotate(
                membrane_kernel,
                angle,
                resize=False,
                center=None,
                order=0,
                mode="edge",
                clip=True,
                preserve_range=True,
            )
        )

    test_images = []
    for test_kernel in test_kernels:
        test_images.append(
            ndimage.filters.convolve(
                image, test_kernel, mode="constant", cval=0
            )
        )

    stacked_images = np.dstack(tuple(test_images))
    membrane_projection_images = {}
    for op in operators:
        membrane_projection_images["membrane_" + op.__name__] = op(
            stacked_images, axis=2
        )

    return membrane_projection_images


def hessian_2d_filter(image):
    """ TODO
    """
    return filters.hessian(image)


def bilateral_2d_filter(image):
    """ TODO
    """
    return restoration.denoise_bilateral(
        image,
        win_size=None,
        sigma_color=None,
        sigma_spatial=1,
        bins=10000,
        mode="constant",
        cval=0,
        multichannel=False,
    )


def gabor_2d_filter(image):
    """ TODO
    For 3D compatibility:
    https://github.com/scikit-image/scikit-image/issues/2704
    """
    gabor_real_image, gabor_imaginary_image = filters.gabor(
        image,
        frequency=3,
        theta=0,
        bandwidth=1,
        sigma_x=None,
        sigma_y=None,
        n_stds=3,
        offset=0,
        mode="reflect",
        cval=0,
    )
    return gabor_real_image


def entropy_2d_filter(image):
    """Computes the entropy around each pixel of an image by looking at
    a disk of radius 3 surrounding it.

    Keyword arguments:
        image -- a 2d-array representing the image to which the filter
        will be applied

    Returns:
        ndarray: an image that is the same size as the original image
        representing the entropy at each pixel in the original image.
    """
    return filters.rank.entropy(image, selem=morphology.disk(3))


def structure_2d_filter(image):
    """Computes the structure tensor around each pixel of an image.

    For 3D compatibability, see here:
    https://github.com/scikit-image/scikit-image/issues/2972

    Keyword arguments:
        image -- The image to which the filter will be applied

    Returns:
        list(ndarray): a list of images that is the same size as the
        original image representing the largest and smallest eigenvalues
        of the structure tensor at each pixel of the original image.
    """
    structure_tensor = feature.structure_tensor(
        image, sigma=1, mode="constant", cval=0
    )
    largest_eig_image = np.zeros(shape=image.shape)
    smallest_eig_image = np.zeros(shape=image.shape)
    for row in range(structure_tensor[0].shape[0]):
        for col in range(structure_tensor[0].shape[1]):
            Axx = structure_tensor[0][row, col]
            Axy = structure_tensor[1][row, col]
            Ayy = structure_tensor[2][row, col]
            eigs = np.linalg.eigvals([[Axx, Axy], [Axy, Ayy]])
            largest_eig_image[row, col] = np.max(eigs)
            smallest_eig_image[row, col] = np.min(eigs)

    return [largest_eig_image, smallest_eig_image]


# Not implemented:


def lipschitz_filter(image):
    """ TODO
    """
    pass


def kuwahara_filter(image):
    """ TODO
    """
    pass


def derivative_filter(image):
    """ TODO
    """
    pass

