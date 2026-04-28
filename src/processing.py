import cv2 # type: ignore
import numpy as np
import multiprocessing as mp
from src.image import CImage

from src.util.plotting import plot_images

class ImageProcessor:
    
    from src.convolution import Convolution
    from src.deconvolution import Deconvolve
    
    @staticmethod
    def apply_kernel_to_image(_image: CImage, _kernel: np.ndarray):

        image_data = _image.data
        imageOutput = cv2.filter2D(image_data, 0, kernel=_kernel)
        imageOutput = cv2.normalize(imageOutput, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cImageOutput = CImage(imageOutput, "updated_" + _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE)
        cImageOutput.save_image(cImageOutput.name)

        imageInput = CImage.load_image(cImageOutput.name)
        imageInput.to_double()

        params = ImageProcessor.Deconvolve.MinRankKernel.MinRankKernelParam()

        image, kernel = ImageProcessor.Deconvolve.deconvolve_image(imageInput, _kernel.shape[0]*2+1, params)

        if params.verbose:
            plot_images({"input image":_image.data, "input kernel":_kernel, "output image": cImageOutput.data, "kernel output": kernel, "image": image}, 3, 2)

        return cImageOutput

    @staticmethod
    def deconvolve_image(_image: CImage, _ksize: int = 9, _params: Deconvolve.MinRankKernel.MinRankKernelParam = None):
        if not _params:
            params = ImageProcessor.Deconvolve.MinRankKernel.MinRankKernelParam()
        else:
            params = _params

        _image.to_double()
        output_cimage, kernel = ImageProcessor.Deconvolve.deconvolve_image(_image, _ksize, params, True)

        if params.verbose:
            plot_images({"input image": _image.data, "kernel": kernel, "output image":output_cimage.data},3,1)

        return output_cimage, kernel