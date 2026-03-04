import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class ImageProcessor:

    class Deconvolve:

        class Param:
            def __init__(self):
                self.tx = 1e-2
                self.tau = 1e-5

                self.delta = 1e-5
                # total x-k iters
                self.imax = 5

                #x step outer iter #
                self.ximax = 2

                # x step inner iter #
                self.xjmax = 2

                # k step kernel  iter #
                self.kmax = 3

                #k step  low rank step #
                self.rmax = 3

                # flag whether using low rank
                self.sigma = 1

                # image reg param
                self.lamb = 80

                # kernel threshold (threshold * max)
                self.threshold = 0.05

                self.mu =1

                # k step total iter #
                self.iterkrank = 10

                # whether show inter results (suggested true to tune...)
                self.verbose = True


        @staticmethod
        def deconvolve_cry(y, K, params: Param):
            assert (np.mod(K,2)==1), "Kernel size must be odd!"

            minscale = int((np.max((2*np.floor((K-1)/32)+1,3))))
            layer = minscale
            i = 1
            step = np.sqrt(2)

            scale = []
            while layer < K:
                scale.append(layer)
                i += 1
                layer = np.floor(layer*step)
                if np.mod(layer,2)== 0:
                    layer += 1

            scale.append(K)

            kernel = np.zeros((minscale, minscale))
            kernel[int(minscale/2), int(minscale/2)] = 0.5
            kernel[int(minscale/2), int(minscale/2)-1] = 0.5

            x1 = np.zeros((minscale, minscale))
            x2 = np.zeros((minscale, minscale))

            for i in range(len(scale)):
                Ki = scale[i]
                print(f'Processing ksize = {Ki}')
                ratio = Ki / K
                hw = (int(np.floor(y.shape[0] * ratio)), int(np.floor(y.shape[1] * ratio)))
                hwi = (hw[0]-1, hw[1]-1)
                smally = cv2.resize(y, hw)
                x1 = cv2.resize(x1, hwi)
                x2 = cv2.resize(x2, hwi)

                if i != 1:
                    kernel = cv2.resize(kernel, (int(Ki), int(Ki)), interpolation=cv2.INTER_CUBIC)

                dx = np.array([[1, -1], [0, 0]])
                dy = np.array([[1, 0], [-1, 0]])

                L2norm = 6 * Ki / scale[0]

                y1 = convolve2d(smally, dx, "valid")
                y2 = convolve2d(smally, dy, "valid")

                x1, x2, kernel = ImageProcessor.Deconvolve.blind_convolve(y1, y2, x1, x2, kernel, L2norm, params)
                y1, x1, kernel = ImageProcessor.Deconvolve.center_kernel_separate(y1, x1, kernel)
                y2, x2, kernel = ImageProcessor.Deconvolve.center_kernel_separate(y2, x2, kernel)
                
                #plt.figure()
                #plt.subplot(221)
                #plt.imshow(x1, cmap="Grays")
                #plt.subplot(222)
                #plt.imshow(x2, cmap="Grays")
                #plt.subplot(223)
                #plt.imshow(kernel, cmap="Grays")
                #plt.show()

            kernel[kernel < np.max(kernel) * params.threshold] = 0
            kernel = kernel / np.sum(kernel)
        
            return y1, kernel
        
        @staticmethod
        def center_kernel_separate(y, x, k):
            mu_x = np.sum((range(1,k.shape[1]+1) * np.sum(k, 0)))
            mu_y = np.sum((range(1,k.shape[0]+1) * np.transpose(np.sum(k,1))))
            
            offset_x = int(np.round( np.floor(k.shape[1] / 2) + 1 - mu_x ))
            offset_y = int(np.round( np.floor(k.shape[0] / 2) + 1 - mu_y ))

            print(f'CenterKernel: weightedMean[{mu_x},{mu_y}] offset[{offset_x},{offset_y}]')

            shift_kernel = np.zeros((np.abs(offset_y * 2) + 1, np.abs(offset_x * 2) + 1))
            shift_kernel[np.abs(offset_y) + offset_y, np.abs(offset_x) + offset_x] = 1
            
            kshift = convolve2d(k, shift_kernel, 'same')
            xshift = convolve2d(x, np.flip(np.flip(shift_kernel,1),0), 'same')
            yshift = convolve2d(y, np.flip(np.flip(shift_kernel,1),0), 'same')

            return yshift, xshift, kshift

        @staticmethod
        def blind_convolve(y1, y2, x1, x2, kernel, L2norm, params: Param):
            if np.sum(np.abs(x1)) == 0:
                x1 = y1
            if np.sum(np.abs(x2)) == 0:
                x2 = y2

            x1 = x1 * L2norm  / ImageProcessor.Deconvolve.norm2(x1)
            x2 = x2 * L2norm  / ImageProcessor.Deconvolve.norm2(x2)

            bhs = int(np.floor(kernel.shape[0] / 2))
            y1v = y1[bhs : y1.shape[0] - bhs, bhs : y1.shape[1] - bhs]
            y2v = y2[bhs : y2.shape[0] - bhs, bhs : y2.shape[1] - bhs]

            #changes = np.zeros((params.imax, 1))
            #k0 = kernel.copy()
            for i in range(params.imax):
                print(f'Iteration {i+1}...', flush=True)
                x1, x2 = ImageProcessor.Deconvolve.optimize_x(x1, x2, kernel, y1, y2, params)
                for iter in range(params.iterkrank):
                    if iter == 0:
                        tmpmu = 0
                    else:
                        tmpmu = params.mu * np.exp(iter) / np.exp(params.iterkrank)
                    
                    kernel = ImageProcessor.Deconvolve.optimize_k(x1, x2, kernel, y1v, y2v, tmpmu, params)

                    if (params.sigma > 0):
                        kernel = ImageProcessor.Deconvolve.optimize_rank(kernel, params)
                    
                    kernel[kernel < 0] = 0
                    kernel = kernel / np.sum(kernel)
                
                if params.threshold:
                    kernel[kernel < np.max(kernel) * params.threshold * i / params.imax] = 0
                else:
                    kernel[kernel < 0] = 0
                kernel = kernel / np.sum(kernel)

            return x1, x2, kernel

        @staticmethod
        def optimize_x(x1, x2, k, y1, y2, params: Param):
            x1 = x1 * 6 / ImageProcessor.Deconvolve.norm2(x1)
            x2 = x2 * 6 / ImageProcessor.Deconvolve.norm2(x2)

            tmp1 = convolve2d(x1, k, 'same') - y1
            tmp2 = convolve2d(x2, k, 'same') - y2
            costLS0 = params.lamb / 2 * (np.transpose(np.concatenate(tmp1)) * np.concatenate(tmp1) + np.transpose(np.concatenate(tmp2)) * np.concatenate(tmp2))
            costR0 =  ImageProcessor.Deconvolve.norm1(x1) / ImageProcessor.Deconvolve.norm2(x1) + ImageProcessor.Deconvolve.norm1(x2) / ImageProcessor.Deconvolve.norm2(x2)
            cost0 = costLS0 + costR0

            tp = 1e-4

            while params.tau > tp:
                x10 = x1.copy()
                x20 = x2.copy()
                for _ in range(params.imax):
                    l21 = ImageProcessor.Deconvolve.norm2(x1)
                    l22 = ImageProcessor.Deconvolve.norm2(x2)
                    for j in range(params.jmax):
                        gradient1 = params.lamb * convolve2d(convolve2d(x1, k, 'same') - y1,np.rot90(k, 2), 'same')
                        gradient2 = params.lamb * convolve2d(convolve2d(x2, k, 'same') - y2,np.rot90(k, 2), 'same')

                        tmp1 = x1 - t * l21 * gradient1
                        tmp2 = x2 - t * l22 * gradient2
                        s1 = np.sign(tmp1)
                        s2 = np.sign(tmp2)
                        x1 = np.matmul(np.max((0, np.abs(tmp1) -  t)), s1)
                        x2 = np.matmul(np.max((0, np.abs(tmp2) -  t)), s2)
                
                tmp1 = convolve2d(x1, k, 'same') - y1
                tmp2 = convolve2d(x2, k, 'same') - y2
                costLS1 = params.lamb / 2 * (np.traspose(np.concatenate(tmp1)) * np.concatenate(tmp1) + np.traspose(np.concatenate(tmp2)) * np.concatenate(tmp2))
                costR1 = ImageProcessor.Deconvolve.norm1(x1) / ImageProcessor.Deconvolve.norm2(x1) + ImageProcessor.Deconvolve.norm1(x2) / ImageProcessor.Deconvolve.norm2(x2)
                cost1 = costLS1 + costR1
                
                if cost1 > 2 * cost0 or np.isnan(cost1):
                    t = t / 2
                    x1 = x10.copy()
                    x2 = x20.copy()
                else:
                    break
            return x1, x2
            
        @staticmethod
        def optimize_k(x1, x2, k, y1, y2, tmpmu, params: Param):
            ep = 1e-3
            i = 0
            k0 = k.copy()
            gradLS1 = 2 * convolve2d(np.rot90(x1, 2), convolve2d(x1, k, 'valid'), 'valid')
            gradLS2 = 2 * convolve2d(np.rot90(x2, 2), convolve2d(x2, k, 'valid'), 'valid')
            Ak = gradLS1 + gradLS2 + 2 * tmpmu * k

            b = 2 * convolve2d(np.rot90(x1, 2), y1, 'valid') + 2 * convolve2d(np.rot90(x2, 2), y2, 'valid')
            b = b + 2 * tmpmu * k0
            r = b - Ak

            d = r.copy()
            e1 = np.matmul(np.transpose(np.concatenate(r)), np.concatenate(r))
            e0 = e1

            while e1 > ep * e0 and i < params.imax:
                gradLS1 = 2 * convolve2d(np.rot90(x1, 2), convolve2d(x1, d, 'valid'), 'valid')
                gradLS2 = 2 * convolve2d(np.rot90(x2, 2), convolve2d(x2, d, 'valid'), 'valid')
                Ad = gradLS1 + gradLS2 + 2 * tmpmu * d; 
                q = Ad.copy()
                alpha = e1 / (np.matmul(np.transpose(np.concatenate(d)), np.concatenate(q)))
                k = k + alpha * d
                
                if not np.mod(i, 50):
                    
                    gradLS1 = 2 * convolve2d(np.rot90(x1, 2), convolve2d(x1, d, 'valid') - y1, 'valid')
                    gradLS2 = 2 * convolve2d(np.rot90(x2, 2), convolve2d(x2, d, 'valid') - y2, 'valid')
                    gradLS = gradLS1 + gradLS2 + 2 * tmpmu * (k - k0)

                    r = -gradLS
                else:
                    r = r - alpha * q
                
                e0 = e1
                e1 = np.matmul(np.transpose(np.concatenate(r)), np.concatenate(r))
                beta = e1 / e0
                d = r + beta * d
                i = i + 1
            
            return k

        @staticmethod
        def optimize_rank(k0, params: Param):
            mu = 1
            Xh = k0.copy()
            w = mu * np.ones((Xh.shape[0],1))

            for _ in range(params.imax):
                Uh, Sh ,Vh = np.linalg.svd(Xh)
                Lh = np.dot(Uh * np.maximum(Sh - (params.tau * np.diag(w)), np.zeros(Sh.shape)), Vh) #
                SLh = np.linalg.svdvals(Lh)
                w = np.ones((w.shape[0],1)) / (SLh+params.delta) 

            return Lh

        @staticmethod
        def norm2(x):
            return np.sqrt(np.sum(np.pow(np.abs(x),2)))

        @staticmethod
        def norm1(x):
            return np.sum(np.abs(x))

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

    
    def load_image(self, _image_path):
        image= cv2.imread(_image_path, 0)#cv2.IMREAD_COLOR_RGB)
        assert image is not None, f"Failed to load image from path: {_image_path}"
        
        self.images[_image_path] = image
        return self.images[_image_path]

    def get_image(self, _image_path):
        if _image_path not in self.images:
            self.load_image(_image_path)
        return self.images[_image_path]

    def calculate_energy(self, _image):
        return np.sum(_image**2)
    
    def calculate_energy_spectral(self, _image):
        return np.sum(ImageProcessor.FourierTransform.compute_image_from_spectrum(_image)**2)
    
    def apply_kernel_to_image(self, _image, _kernel):

        #imageSpectrum = ImageProcessor.FourierTransform.compute_spectrum_from_image(_image)

        #rows, columns = _image.shape
        #mask = ImageProcessor.KernelFilters.convert_kernel_to_mask(_kernel, _image.shape)

        #imageFiltered = imageSpectrum * mask

        #imageOutput = ImageProcessor.FourierTransform.compute_image_from_spectrum(imageFiltered)

        imageOutput = cv2.filter2D(_image, 0, kernel=_kernel)

        imageFilteredRead = ImageProcessor.FourierTransform.compute_spectrum_from_image(imageOutput)

        params = self.Deconvolve.Param()
        image, kernel = self.Deconvolve.deconvolve_cry(imageOutput.astype(np.double), 19, params)

        if self.verbose:
            plt.figure()
            plt.subplot(331)
            plt.imshow(_image, cmap='gray')
            plt.subplot(332)
            plt.imshow(_kernel, cmap='gray')
            #plt.subplot(333)
            #plt.imshow(ImageProcessor.FourierTransform.compute_spectrum_decibel_image(mask), cmap='gray') 
            #plt.subplot(334)
            #plt.imshow(ImageProcessor.FourierTransform.compute_spectrum_decibel_image(imageFiltered), cmap='gray')  
            plt.subplot(335)
            plt.imshow(imageOutput, cmap='gray')
            plt.subplot(336)
            plt.imshow(ImageProcessor.FourierTransform.compute_spectrum_decibel_image(imageFilteredRead), cmap='gray')
            plt.subplot(337)
            plt.imshow(image, cmap='gray')
            plt.subplot(338)
            plt.imshow(kernel, cmap='gray')
            plt.show()

        return cv2.normalize(imageOutput, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

    def deconvolve_image(self, _image):

        params = self.Deconvolve.Param()
        image, kernel = self.Deconvolve.deconvolve_cry(_image.astype(np.double), 55, params)
        

        if self.verbose:
            plt.figure()
            plt.subplot(221)
            plt.imshow(image, cmap='gray')
            plt.subplot(222)
            plt.imshow(image, cmap='gray')
            plt.subplot(223)
            plt.imshow(kernel, cmap='gray')
            plt.show()

        return image

    def process_image(self):
        # Example processing: Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def save_image(self, _image, _output_path):
        cv2.imwrite(_output_path, _image)