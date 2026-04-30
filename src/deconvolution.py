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
        end_profiling(profile, f"profiling/{_image.name}_{str(_ksize)}temp.txt")

        return kernel

    @staticmethod
    def deconvolve_cimage_bregman(_image: CImage, _kernel: np.ndarray, _verbose=False):
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
            output_image, _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE, time_stamp=True
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
    def deconvolve_image(_image: CImage, _kernel_size: int = -1):
        kernel_size = _kernel_size if _kernel_size != -1 else 9
        param = MinRankKernel.MinRankKernelParam()
        kernel = Deconvolve.deconvolve_cimage_mrk(_image, kernel_size, param)
        output_cimage = Deconvolve.deconvolve_cimage_bregman(
            _image, kernel, _verbose=True
        )
        return output_cimage

    @staticmethod
    def convert_to_ycbcr(_image: CImage):
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
