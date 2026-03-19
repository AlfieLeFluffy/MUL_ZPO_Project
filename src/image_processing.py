import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from src.image import CImage

from src.deconvolution import Deconvolve

class ImageProcessor:

   
    class FourierTransform:
        @staticmethod
        def compute_dft(_image):
            return cv2.dft(np.float32(_image), flags=cv2.DFT_COMPLEX_OUTPUT)

        @staticmethod
        def compute_idft(_dft_image):
            return cv2.idft(_dft_image, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        @staticmethod
        def shift_dft(_dft_image):
            return np.fft.fftshift(_dft_image)
        
        @staticmethod
        def unshift_dft(_dft_image):
            return np.fft.ifftshift(_dft_image)

        @staticmethod
        def compute_spectrum_decibel_image(_dft_image):
            if _dft_image.ndim == 3:
                return 20 * np.log(cv2.magnitude(_dft_image[:,:,0], _dft_image[:,:,1]) + np.finfo(float).eps)
            return 20 * np.log(_dft_image + np.finfo(float).eps)
        
        @staticmethod
        def compute_magnitude(_dft_image):
            return cv2.magnitude(_dft_image[:,:,0], _dft_image[:,:,1])
        
        @staticmethod
        def compute_spectrum_from_image(_image):
            imageDFT = ImageProcessor.FourierTransform.compute_dft(_image)
            shifted = ImageProcessor.FourierTransform.shift_dft(imageDFT)
            return shifted

        @staticmethod
        def compute_image_from_spectrum(_dft_image):
            unshifted = ImageProcessor.FourierTransform.unshift_dft(_dft_image)
            undft = ImageProcessor.FourierTransform.compute_idft(unshifted)
            return undft
        
        @staticmethod
        def apply_filter_to_image(_image, _filter):
            imageDFT = ImageProcessor.FourierTransform.compute_spectrum_from_image(_image)
            filtered = imageDFT * _filter
            imageOutput = ImageProcessor.FourierTransform.compute_image_from_spectrum(filtered)
            return imageOutput
        
        @staticmethod
        def reverse_filter_to_image(_image, _filter):
            imageDFT = ImageProcessor.FourierTransform.compute_spectrum_from_image(_image)
            filtered = imageDFT / _filter
            imageOutput = ImageProcessor.FourierTransform.compute_image_from_spectrum(filtered)
            return imageOutput

    class KernelFilters:
        @staticmethod
        def create_low_pass_filter(_kernel_size, _offset=50):
            kernel = np.zeros(_kernel_size, dtype=np.float32)
            centerX = _kernel_size[0] // 2
            centerY = _kernel_size[1] // 2
            if len(_kernel_size) == 3:
                for i in range(kernel.shape[2]):
                    kernel[:,:,i] = ImageProcessor.KernelFilters.create_low_pass_filter((_kernel_size[0], _kernel_size[1]), _offset)
            else:
                kernel[centerX-_offset:centerX+_offset, centerY-_offset:centerY+_offset] = 1
                kernel = kernel / np.sum(kernel)
            return kernel
        
        @staticmethod
        def create_high_pass_filter(_kernel_size, _offset=50):
            kernel = np.ones(_kernel_size, dtype=np.float32)
            centerX = _kernel_size[0] // 2
            centerY = _kernel_size[1] // 2
            if len(_kernel_size) == 3:
                for i in range(kernel.shape[2]):
                    kernel[:,:,i] = ImageProcessor.KernelFilters.create_high_pass_filter((_kernel_size[0], _kernel_size[1]), _offset)
            else:
                kernel[centerX-_offset:centerX+_offset, centerY-_offset:centerY+_offset] = 0
            return kernel
        
        @staticmethod
        def create_hamming_window(_kernel_size):
            if len(_kernel_size) == 3:
                kernel = np.zeros(_kernel_size, dtype=np.float32)
                for i in range(kernel.shape[2]):
                    kernel[:,:,i] = ImageProcessor.KernelFilters.create_hamming_window((_kernel_size[0], _kernel_size[1]))
                return kernel
            else:
                return np.sqrt(np.outer(np.hamming(_kernel_size[0]), np.hamming(_kernel_size[1])))

        @staticmethod
        def convert_kernel_to_mask(_kernel, _image_size):
            rows, columns = _image_size
            mask = np.dstack((_kernel, _kernel))
            mask = cv2.dft(mask, flags=cv2.DFT_COMPLEX_OUTPUT, dst=_kernel)
            mask = np.fft.fftshift(mask)
            mask = cv2.resize(mask, (columns, rows))
            return mask

    def __init__(self, _verbose=True):
        self.images = {}
        self.verbose = _verbose

    def get_image(self, _image_path):
        if _image_path not in self.images:
            self.load_image(_image_path)
        return self.images[_image_path]

    def calculate_energy(self, _image):
        return np.sum(_image**2)
    
    def calculate_energy_spectral(self, _image):
        return np.sum(ImageProcessor.FourierTransform.compute_image_from_spectrum(_image)**2)
    
    def apply_kernel_to_image(self, _image: CImage, _kernel):

        #imageSpectrum = ImageProcessor.FourierTransform.compute_spectrum_from_image(_image)

        #rows, columns = _image.shape
        #mask = ImageProcessor.KernelFilters.convert_kernel_to_mask(_kernel, _image.shape)

        #imageFiltered = imageSpectrum * mask

        #imageOutput = ImageProcessor.FourierTransform.compute_image_from_spectrum(imageFiltered)
        image_data = _image.data
        imageOutput = cv2.filter2D(image_data, 0, kernel=_kernel)
        imageOutput = cv2.normalize(imageOutput, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        imageOutputC = CImage(imageOutput, "updated_" + _image.name, CImage.IMAGE_TYPE.YCBCR_DOUBLE)
        imageOutputC.save_image(imageOutputC.name)

        imageInput = CImage.load_image(imageOutputC.name)
        imageInput.to_double()

        #imageFilteredRead = ImageProcessor.FourierTransform.compute_spectrum_from_image(imageOutput)

        params = Deconvolve.MinRankKernel.Param()

        image, kernel = Deconvolve.deconvlove_image(imageInput, 9, params)
        #kernel = Deconvolve.MinRankKernel.deconvolve_cry(imageOutput.astype(np.double), 9, params)

        if self.verbose:
            plt.figure()
            plt.subplot(231)
            plt.imshow(_image.data)
            plt.subplot(232)
            plt.imshow(_kernel)  
            plt.subplot(233)
            plt.imshow(imageOutputC.data)
            plt.subplot(235)
            plt.imshow(kernel)
            plt.subplot(236)
            plt.imshow(image)
            plt.show()

        return imageOutputC

    def deconvolve_image(self, _image):

        params = Deconvolve.MinRankKernel.Param()
        image, kernel = Deconvolve.deconvlove_image(_image.astype(np.double), 85, params, False)

        if self.verbose:
            plt.figure()
            plt.imshow(kernel)
            plt.show()

        return kernel

    def process_image(self):
        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return gray_image