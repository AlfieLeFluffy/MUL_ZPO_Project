import cv2
import numpy as np

class SpectralConvolution:
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
            imageDFT = SpectralConvolution.compute_dft(_image)
            shifted = SpectralConvolution.shift_dft(imageDFT)
            return shifted

        @staticmethod
        def compute_image_from_spectrum(_dft_image):
            unshifted = SpectralConvolution.unshift_dft(_dft_image)
            undft = SpectralConvolution.compute_idft(unshifted)
            return undft
        
        @staticmethod
        def apply_filter_to_image(_image, _filter):
            imageDFT = SpectralConvolution.compute_spectrum_from_image(_image)
            filtered = imageDFT * _filter
            imageOutput = SpectralConvolution.compute_image_from_spectrum(filtered)
            return imageOutput
        
        @staticmethod
        def reverse_filter_to_image(_image, _filter):
            imageDFT = SpectralConvolution.compute_spectrum_from_image(_image)
            filtered = imageDFT / _filter
            imageOutput = SpectralConvolution.compute_image_from_spectrum(filtered)
            return imageOutput
