import cv2  # type: ignore
import numpy as np

# import multiprocessing as mp

from src.image import CImage
from src.util.plotting import plot_images


class ImageProcessor:
    from src.convolution import Convolution
    from src.deconvolution import Deconvolve

    @staticmethod
    def filter_cimage(_image: CImage, _kernel: np.ndarray):
        """Applies a specified kernel to an image.

        Args:
            _image (CImage): Input image
            _kernel (np.ndarray): Input kernel

        Returns:
            CImage: Output image
        """

        image_data = _image.data
        imageOutput = cv2.filter2D(image_data, 0, kernel=_kernel)
        imageOutput = cv2.normalize(
            imageOutput, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        cImageOutput = CImage(
            imageOutput, _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE, time_stamp=True
        )

        return cImageOutput

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

        _image.to_double()
        kernel = ImageProcessor.Deconvolve.deconvolve_cimage_mrk(_image, _ksize, params)
        output_cimage = ImageProcessor.Deconvolve.deconvolve_cimage_bregman(
            _image, kernel, _verbose=params.verbose
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
