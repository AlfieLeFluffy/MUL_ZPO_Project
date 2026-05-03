import cv2  # type: ignore
import numpy as np

from src.image import CImage
from src.util.plotting import plot_images
from src.conv.spectral_convolution import SpectralConvolution as sc


class ImageProcessor:
    from src.convolution import Convolution
    from src.deconvolution import Deconvolve

    @staticmethod
    def convolve(_image: CImage, _kernel: np.ndarray, _verbose: bool = False):
        """Applies a specified kernel to an image.

        Args:
            _image (CImage): Input image
            _kernel (np.ndarray): Input kernel

        Returns:
            CImage: Output image
        """

        image_data = _image.data
        output = sc.fft_convolve(image_data, _kernel)
        output = output.astype(np.uint8)
        if _verbose:
            plot_images(
                {
                    "image": image_data,
                    "kernel": _kernel,
                    "output": output,
                },
                2,
                2,
            )
        return CImage(output, _image.name, CImage.IMAGE_TYPE.RGB_INT, _stamp=True)

    @staticmethod
    def deconvolve_cimage(
        _image: CImage,
        _ksize: int = 9,
        _params: Deconvolve.MinRankKernel.MinRankKernelParam = None,
    ):
        """Takes in an image and deconvolves it using the Min-Rank-Kernel and Bregman deconvolution algorithms.

        Args:
            _image (CImage): Input image
            _ksize (int, optional): Expected kernel size. Defaults to 9.
            _params (Deconvolve.MinRankKernel.MinRankKernelParam, optional): Min-Rank-Kernel algorithm parameters. Defaults to None.

        Returns:
            (CImage, np.ndarray): Output processed image, Found kernel
        """
        if not _params:
            params = ImageProcessor.Deconvolve.MinRankKernel.MinRankKernelParam()
        else:
            params = _params

        image_double = _image.to_double()
        kernel = ImageProcessor.Deconvolve.deconvolve_cimage_mrk(
            image_double, _ksize, params
        )
        output_cimage = ImageProcessor.Deconvolve.deconvolve_cimage_bregman(
            image_double, kernel, _verbose=params.verbose
        )

        if params.verbose:
            plot_images(
                {
                    "input image": _image.data,
                    "kernel": kernel,
                    "output image": output_cimage.data,
                },
                3,
                1,
            )

        return output_cimage, kernel

    @staticmethod
    def grayscale_cimage(
        _image: CImage,
    ):
        """Converts a CImage instance to grayscale.

        Args:
            _image (CImage): Input image

        Returns:
            CImage: Grayscale version of the input image
        """
        if _image.type in [CImage.IMAGE_TYPE.RGB_INT, CImage.IMAGE_TYPE.BGR_INT]:
            grayscale_data = cv2.cvtColor(_image.data, cv2.COLOR_RGB2GRAY)
            image_type = CImage.IMAGE_TYPE.RGB_INT
        elif _image.type in [
            CImage.IMAGE_TYPE.RGB_DOUBLE,
            CImage.IMAGE_TYPE.BGR_DOUBLE,
        ]:
            grayscale_data = cv2.cvtColor(_image.data, cv2.COLOR_RGB2GRAY)
            image_type = CImage.IMAGE_TYPE.RGB_DOUBLE
        elif _image.type in [
            CImage.IMAGE_TYPE.YCBCR_INT,
            CImage.IMAGE_TYPE.YCBCR_DOUBLE,
        ]:
            grayscale_data = _image.data[:, :, 0]
            image_type = _image.type
        else:
            raise ValueError(f"Unknown image type: {_image.type}")

        output_cimage = CImage(grayscale_data, _image.name, image_type, _stamp=True)
        return output_cimage
