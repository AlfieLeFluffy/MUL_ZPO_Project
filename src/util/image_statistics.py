import numpy as np

from src.conv.spectral_convolution import SpectralConvolution


@staticmethod
def calculate_energy(_image):
    return np.sum(_image**2)


@staticmethod
def calculate_energy_spectral(_image):
    return np.sum(SpectralConvolution.spec2image(_image) ** 2)
