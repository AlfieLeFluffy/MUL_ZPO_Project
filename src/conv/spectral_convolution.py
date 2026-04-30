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
    def shift_spec(_dft_image):
        return np.fft.fftshift(_dft_image)

    @staticmethod
    def unshift_spec(_dft_image):
        return np.fft.ifftshift(_dft_image)

    @staticmethod
    def spec_decibel(_dft_image):
        if _dft_image.ndim == 3:
            return 20 * np.log(
                cv2.magnitude(_dft_image[:, :, 0], _dft_image[:, :, 1])
                + np.finfo(float).eps
            )
        return 20 * np.log(_dft_image + np.finfo(float).eps)

    @staticmethod
    def image_magnitude(_dft_image):
        return cv2.magnitude(_dft_image[:, :, 0], _dft_image[:, :, 1])

    @staticmethod
    def image2spec(_image):
        imageDFT = SpectralConvolution.compute_dft(_image)
        shifted = SpectralConvolution.shift_spec(imageDFT)
        return shifted

    @staticmethod
    def spec2image(_dft_image):
        unshifted = SpectralConvolution.unshift_spec(_dft_image)
        undft = SpectralConvolution.compute_idft(unshifted)
        return undft

    @staticmethod
    def filter_image(_image, _filter):
        imageDFT = SpectralConvolution.image2spec(_image)
        filtered = imageDFT * _filter
        imageOutput = SpectralConvolution.spec2image(filtered)
        return imageOutput

    @staticmethod
    def filter_image_reverse(_image, _filter):
        imageDFT = SpectralConvolution.image2spec(_image)
        filtered = imageDFT / _filter
        imageOutput = SpectralConvolution.spec2image(filtered)
        return imageOutput
