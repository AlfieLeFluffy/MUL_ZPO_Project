import numpy as np

from src.image import CImage
from src.deconv.min_rank_kernel import MinRankKernel
from src.util.image_preprocessing import edgetaper

from src.util.profiling import start_profiling, end_profiling
from src.util.plotting import plot_images


class Deconvolve:
    """
    Collection of deconvolution algorithms.
    """

    from src.deconv.min_rank_kernel import MinRankKernel
    from src.deconv.bregman import Bregman

    @staticmethod
    def deconvolve_cimage_mrk(
        _image: CImage,
        _ksize,
        _params: MinRankKernel.MinRankKernelParam,
    ):
        """Takes in an input image, expected kernel size and parameters, which are then used to estimate the kernel by which the image was affected.

        Args:
            _image (CImage): Input image
            _ksize (int): Estimated size of the kernel
            _params (MinRankKernel.MinRankKernelParam): MRK parameters

        Raises:
            Exception: _description_

        Returns:
            np.ndarray: Estimated kernel
        """
        ycbcr = Deconvolve.convert_to_ycbcr(_image)

        profile = start_profiling()
        kernel = Deconvolve.MinRankKernel.deconvolve(
            ycbcr.data[:, :, 0], _ksize, _params
        )
        end_profiling(profile, f"{_image.name}_{str(_ksize)}-temp.log")

        return kernel

    @staticmethod
    def deconvolve_cimage_bregman(_image: CImage, _kernel: np.ndarray, _verbose=False):
        """Deconvolution method using the split fast Bregman method.

        Args:
            _image (CImage): Input image
            _kernel (np.ndarray): Input kernel
            _verbose (bool, optional): Verbose. Defaults to False.

        Returns:
            CImage: Output image
        """
        ycbcr = Deconvolve.convert_to_ycbcr(_image)

        nb_lambda = 3000
        nb_alpha = 1
        bhs = int(np.floor(_kernel.shape[0]))
        ypad = np.pad(ycbcr.data[:, :, 0], bhs, "edge")

        for _ in range(3):
            ypad = edgetaper(ypad, _kernel)

        bregman_decon = Deconvolve.Bregman()
        output_y = bregman_decon.deconvolve(
            ypad, _kernel, nb_lambda, nb_alpha, _verbose=_verbose
        )
        output_image = ycbcr.data.copy()
        output_image[:, :, 0] = output_y[
            bhs : output_y.shape[0] - bhs, bhs : output_y.shape[1] - bhs
        ]

        output_cimage = CImage(
            output_image, _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE, _stamp=True
        )
        output_cimage = output_cimage.ycbcr2rgb()

        if _verbose:
            plot_images(
                {
                    "original image": _image.data,
                    "ycbcr image": ycbcr.data,
                    "padded image": ypad,
                    "processed image": output_cimage.data,
                },
                2,
                2,
            )

        return output_cimage

    @staticmethod
    def deconvolve_cimage_direct(_image: CImage, _kernel: np.ndarray):
        """Testing brute force method that directly divides the image spectrum by the kernel.

        Args:
            _image (CImage): Input image
            _kernel (np.ndarray): Input kernel

        Raises:
            ValueError: Many

        Returns:
            CImage: Output image
        """
        data = _image.data
        image = np.asarray(data, dtype=float)
        kernel = np.asarray(_kernel, dtype=float)

        if image.ndim == 3 and kernel.ndim == 2:
            channels = [
                Deconvolve.deconvolve_cimage_direct(image[:, :, c], kernel)
                for c in range(image.shape[2])
            ]
            return CImage(
                np.stack(channels, axis=2).astype(np.uint8),
                _image.name,
                _stamp=True,
            )

        if image.ndim != 2 or kernel.ndim != 2:
            raise ValueError(
                "fft_convolve supports a 2D image and 2D kernel, or a 3D image with a 2D kernel"
            )

        fft_shape = tuple(np.array(image.shape) + np.array(kernel.shape) - 1)
        image_fft = np.fft.fft2(image, s=fft_shape)
        kernel_fft = np.fft.fft2(kernel, s=fft_shape)
        deconv = np.fft.ifft2(image_fft / kernel_fft)
        deconv = np.real(deconv)

        return deconv

    @staticmethod
    def deconvolve_image(_image: CImage, _kernel_size: int = -1):
        """A method that combines both MRK and Bregman into one function call.

        Args:
            _image (CImage): Input image
            _kernel_size (int, optional): Expected kernel size. Defaults to -1.

        Returns:
            CImage: Ouput image
        """
        kernel_size = _kernel_size if _kernel_size != -1 else 9
        param = MinRankKernel.MinRankKernelParam()
        kernel = Deconvolve.deconvolve_cimage_mrk(_image, kernel_size, param)
        output_cimage = Deconvolve.deconvolve_cimage_bregman(
            _image, kernel, _verbose=True
        )
        return output_cimage

    @staticmethod
    def convert_to_ycbcr(_image: CImage):
        """A function that check and converts a CImage into a YYCBCR CImage

        Args:
            _image (CImage): Input image

        Raises:
            Exception: Hopefully nonw

        Returns:
            CImage: Ouput YCBCR CImage
        """
        if _image.type in [CImage.IMAGE_TYPE.RGB_INT, CImage.IMAGE_TYPE.RGB_DOUBLE]:
            ycbcr: CImage = _image.rgb2ycbcr()
        elif _image.type in [
            CImage.IMAGE_TYPE.YCBCR_INT,
            CImage.IMAGE_TYPE.YCBCR_DOUBLE,
        ]:
            ycbcr = _image
        else:
            raise Exception("What is this picture?")

        return ycbcr
