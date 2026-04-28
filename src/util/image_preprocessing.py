import cv2
import numpy as np

@staticmethod
def edgetaper(img, psf):
    """
    Taper edges of an image to reduce DFT ringing artifacts.
    """
    h, w = img.shape
    
    # 1. Create a tapering mask (alpha) based on the PSF
    # A simple approach is a border that fades out
    alpha = np.ones((h, w), dtype=np.float32)
    
    # Create a margin based on PSF size
    margin_y = int(psf.shape[0] * 2)
    margin_x = int(psf.shape[1] * 2)
    
    # Apply a gentle fade (linear or gaussian) at the borders
    for i in range(min(margin_y, h//2)):
        alpha[i, :] *= (i / margin_y)
        alpha[h-1-i, :] *= (i / margin_y)
    for i in range(min(margin_x, w//2)):
        alpha[:, i] *= (i / margin_x)
        alpha[:, w-1-i] *= (i / margin_x)
    
    # Smooth the mask to avoid sharp, artificial transitions
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    # 2. Blur the original image to create a "background" for the edges
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    
    # 3. Blend the original and blurred image using the mask
    # Need to handle color channels
    if len(img.shape) == 3:
        alpha = cv2.merge([alpha, alpha, alpha])
        
    tapered_img = alpha * img + (1.0 - alpha) * blurred
    
    return tapered_img

@staticmethod
def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros

    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image

    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.all(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                            "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


@staticmethod
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array

    Returns
    -------
    otf : `numpy.ndarray`
        OTF array

    Notes
    -----
    Adapted from MATLAB psf2otf function
    Arrays of higher dimension that 2D are also supported

    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fftn(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

@staticmethod
def otf2psf(otf, shape=None):
    if shape is None:
        shape = otf.shape
    
    # Inverse Fourier transform
    psf = np.fft.ifftn(otf, shape)
    
    # Shift the zero-frequency component to the center
    psf = np.fft.fftshift(psf)
    
    # Take the real part, as the PSF should be real-valued
    psf = np.real(psf)
    
    # Normalize the PSF so that its sum is 1
    psf /= np.sum(psf)
    
    return psf