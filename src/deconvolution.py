import numpy as np
import matplotlib.pyplot as plt

from src.image import CImage
from src.util.image_preprocessing import edgetaper
from src.util.profiling import start_profiling, end_profiling
from src.util.plotting import plot_images

class Deconvolve:
    """
    Collection of deconvolution algorithms.
    """

    from src.deconv.min_rank_kernel import MinRankKernel
    from src.deconv.bregman import Bregman

    #def conv2_center(_image, _kernel, _shape):
    #    return np.fft.irfft2(np.fft.rfft2(_image, _shape) * np.fft.rfft2(_kernel, _shape), _shape)

    @staticmethod
    def deconvolve_image(_image: CImage, _ksize, _params: MinRankKernel.MinRankKernelParam, _finalization=True):
        nb_lambda = 3000
        nb_alpha = 0.5
        use_ycbcr = True

        if use_ycbcr:
            if len(_image.size) == 3:
                if _image.size[2] == 3:
                    ycbcr: CImage = _image.rgb2ycbcr()
                else:
                    raise Exception("What is this picture?")
            else:
                ycbcr = _image
            nb_alpha = 1
        
        profile = start_profiling()
        kernel = Deconvolve.MinRankKernel.deconvolve(ycbcr.data[:,:,0], _ksize, _params)
        end_profiling(profile, f"profiling/{_image.name}_{str(_ksize)}temp.txt")
        
        if _finalization:
            bhs = int(np.floor(kernel.shape[0]))
            ypad = np.pad(ycbcr.data[:,:,0], bhs, "edge")
            
            for _ in range(3):
                ypad = edgetaper(ypad, kernel)

            bregmain_decon = Deconvolve.Bregman()
            output_y = bregmain_decon.bregman_deconvolution(ypad, kernel, nb_lambda, nb_alpha, _verbose=_params.verbose)
            output_image = ycbcr.data.copy()
            output_image[:,:,0] = output_y[bhs:output_y.shape[0]-bhs,bhs:output_y.shape[1]-bhs]

            output_cimage = CImage(output_image, "decon" + _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE)
            output_cimage = output_cimage.ycbcr2rgb()

            if _params.verbose:
                plot_images({"original image": _image.data, "ycbcr image": ycbcr.data, "padded image": ypad, "processed image": output_cimage.data},2,2)
        else:
            output_cimage = CImage(ycbcr, "nondecon" + _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE)
            output_cimage = output_cimage.ycbcr2rgb()

        return output_cimage, kernel