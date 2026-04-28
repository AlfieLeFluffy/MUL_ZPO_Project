import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from ..util.math import norm1, norm2

class MinRankKernel:
        """
        Blind deconvolution algorithm that can extract kernels of affected images.
        """

        class MinRankKernelParam:
            """
            Parameter class for the Min-Rank-Kernel algorithms.
            """
            def __init__(self):
                self.tx = 1e-2
                self.tau = 1e-5 #3e-4 #1e-5

                self.delta = 1e-5
                # total x-k iters
                self.imax = 5

                #x step outer iter
                self.ximax = 2

                # x step inner iter
                self.xjmax = 2

                # k step kernel  iter
                self.kmax = 3

                #k step  low rank step
                self.rmax = 3

                # flag whether using low rank
                self.sigma = True

                # image reg param
                self.lamb = 80

                # kernel threshold (threshold * max)
                self.threshold = 0.05
                self.mu = 1

                # k step total iter
                self.iterkrank = 10

                # whether show inter results (suggested true to tune...)
                self.verbose = True

        @staticmethod
        def deconvolve(_image, _ksize, params: MinRankKernelParam):
            assert (np.mod(_ksize,2)==1), "Kernel size must be odd!"

            minscale = int((np.max((2*np.floor((_ksize-1)/32)+1,3))))
            layer = minscale
            i = 1
            step = np.sqrt(2)

            scale = []
            while layer < _ksize:
                scale.append(layer)
                i += 1
                layer = int(np.floor(layer*step))
                if np.mod(layer,2)== 0:
                    layer += 1
            scale.append(_ksize)

            kernel = np.zeros((minscale, minscale))
            kcenter = int(minscale/2)
            kernel[kcenter, kcenter] = 0.5
            kernel[kcenter, kcenter-1] = 0.5

            x1 = np.zeros((minscale, minscale))
            x2 = np.zeros((minscale, minscale))

            for i in range(len(scale)):
                Ki = scale[i]
                print(f'Processing ksize = {Ki}')
                ratio = Ki / _ksize
                hw = (int(np.floor(_image.shape[1] * ratio)), int(np.floor(_image.shape[0] * ratio)))
                hwi = (hw[0]-1, hw[1]-1)
                smally = cv2.resize(_image, hw, interpolation=cv2.INTER_AREA).astype(np.double)
                x1 = cv2.resize(x1, hwi, interpolation=cv2.INTER_AREA).astype(np.double)
                x2 = cv2.resize(x2, hwi, interpolation=cv2.INTER_AREA).astype(np.double)

                if i != 0:
                    kernel = cv2.resize(kernel, (int(Ki), int(Ki)), interpolation=cv2.INTER_LINEAR)

                dx = np.array([[1, -1], [0, 0]])
                dy = np.array([[1, 0], [-1, 0]])

                L2norm = 6 * Ki / scale[0]

                y1 = fftconvolve(smally, dx, "valid")
                y2 = fftconvolve(smally, dy, "valid")

                x1, x2, kernel = MinRankKernel.blind_deconvolve(y1, y2, x1, x2, kernel, (params.tau * ((i+1) / len(scale))), L2norm, params)
                y1, x1, kernel = MinRankKernel.center_kernel_separate(y1, x1, kernel, params)
                y2, x2, kernel = MinRankKernel.center_kernel_separate(y2, x2, kernel, params)
                
                if False:
                    plt.figure()
                    plt.subplot(231)
                    plt.title("x1")
                    plt.imshow(x1)
                    plt.subplot(232)
                    plt.title("x2")
                    plt.imshow(x2)
                    plt.subplot(234)
                    plt.title("y1")
                    plt.imshow(y1)
                    plt.subplot(235)
                    plt.title("y2")
                    plt.imshow(y2)
                    plt.subplot(236)
                    plt.title("kernel")
                    plt.imshow(kernel)
                    plt.show()

            kernel[kernel < np.max(kernel) * params.threshold] = 0
            kernel = kernel / np.sum(kernel)
        
            return kernel
        
        @staticmethod
        def center_kernel_separate(y, x, k, params):
            k_shape = k.shape
            mu_y = np.sum((range(1,k_shape[0]+1) * np.sum(k,1).T))
            mu_x = np.sum((range(1,k_shape[1]+1) * np.sum(k,0)))
            
            offset_x = int(np.round( np.floor(k.shape[1] / 2) + 1 - mu_x ))
            offset_y = int(np.round( np.floor(k.shape[0] / 2) + 1 - mu_y ))

            if params.verbose:
                print(f'CenterKernel: weightedMean[{mu_x-1},{mu_y-1}] offset[{offset_x},{offset_y}]')

            shift_kernel = np.zeros((np.abs(offset_y * 2) + 1, np.abs(offset_x * 2) + 1))
            shift_kernel[np.abs(offset_y) + offset_y, np.abs(offset_x) + offset_x] = 1
            
            kshift = fftconvolve(k, shift_kernel, "same")
            xshift = fftconvolve(x, np.flip(np.flip(shift_kernel,0),1), "same")
            yshift = fftconvolve(y, np.flip(np.flip(shift_kernel,0),1), "same")

            return yshift, xshift, kshift

        @staticmethod
        def blind_deconvolve(y1, y2, x1, x2, kernel, tau, L2norm, params: MinRankKernelParam):
            if np.sum(np.abs(x1)) == 0:
                x1 = y1
            if np.sum(np.abs(x2)) == 0:
                x2 = y2

            x1 = x1 * L2norm  / norm2(x1)
            x2 = x2 * L2norm  / norm2(x2)

            ksz = kernel.shape[0]
            bhs = int(np.floor(ksz / 2))
            y1v = y1[bhs : y1.shape[0] - bhs, bhs : y1.shape[1] - bhs]
            y2v = y2[bhs : y2.shape[0] - bhs, bhs : y2.shape[1] - bhs]

            #changes = np.zeros((params.imax, 1))
            #k0 = kernel.copy()
            for i in range(params.imax):
                if params.verbose:
                    print(f'Iteration {i+1}...', flush=True)
                x1, x2 = MinRankKernel.optimize_x(x1, x2, kernel, y1, y2, params)
                for iter in range(params.iterkrank):
                    if iter == 0:
                        tmpmu = 0
                    else:
                        tmpmu = params.mu * np.exp(iter) / np.exp(params.iterkrank)
                    
                    kernel = MinRankKernel.optimize_k(x1, x2, kernel, y1v, y2v, tmpmu, params)

                    if params.sigma:
                        kernel = MinRankKernel.optimize_rank(kernel, tau, params)
                    
                    kernel[kernel < 0] = 0
                    kernel = kernel / np.sum(kernel)

                    if False:
                        plt.figure()
                        plt.subplot(231)
                        plt.title("x1")
                        plt.imshow(x1)
                        plt.subplot(232)
                        plt.title("x2")
                        plt.imshow(x2)
                        plt.subplot(234)
                        plt.title("y1")
                        plt.imshow(y1)
                        plt.subplot(235)
                        plt.title("y2")
                        plt.imshow(y2)
                        plt.subplot(236)
                        plt.title("kernel")
                        plt.imshow(kernel, cmap="gray")
                        plt.show()
                
                if params.threshold:
                    kernel[kernel < (np.max(kernel) * params.threshold * ((i+1) / (params.imax+1)))] = 0
                else:
                    kernel[kernel < 0] = 0
                kernel = kernel / np.sum(kernel)

            return x1, x2, kernel

        @staticmethod
        def optimize_x(_x1, _x2, _k, _y1, _y2, params: MinRankKernelParam):
            x1 = _x1 * 6 / norm2(_x1)
            x2 = _x2 * 6 / norm2(_x2)

            tmp1 = fftconvolve(x1, _k, 'same') - _y1
            tmp2 = fftconvolve(x2, _k, 'same') - _y2
            costLS0 = params.lamb / 2 * (np.matmul(np.concatenate(tmp1).T, np.concatenate(tmp1)) + np.matmul(np.concatenate(tmp2).T, np.concatenate(tmp2)))
            costR0 =  norm1(x1) / norm2(x1) + norm1(x2) / norm2(x2)
            cost0 = costLS0 + costR0

            t = params.tx
            tp = 1e-4

            while t > tp:
                x10 = x1.copy()
                x20 = x2.copy()
                for _ in range(params.ximax):
                    l21 = norm2(x1)
                    l22 = norm2(x2)
                    for _ in range(params.xjmax):
                        gradient1 = params.lamb * fftconvolve(fftconvolve(x1, _k, 'same') - _y1, np.rot90(_k, 2), 'same')
                        gradient2 = params.lamb * fftconvolve(fftconvolve(x2, _k, 'same') - _y2, np.rot90(_k, 2), 'same')

                        tmp1 = x1 - t * l21 * gradient1
                        tmp2 = x2 - t * l22 * gradient2
                        x1 = np.maximum(np.zeros(tmp1.shape), np.abs(tmp1) -  t) * np.sign(tmp1)
                        x2 = np.maximum(np.zeros(tmp2.shape), np.abs(tmp2) -  t) * np.sign(tmp2)
                
                tmp1 = fftconvolve(x1, _k, 'same') - _y1
                tmp2 = fftconvolve(x2, _k, 'same') - _y2
                costLS1 = params.lamb / 2 * (np.matmul(np.concatenate(tmp1).T, np.concatenate(tmp1)) + np.matmul(np.concatenate(tmp2).T, np.concatenate(tmp2)))
                costR1 = norm1(x1) / norm2(x1) + norm1(x2) / norm2(x2)
                cost1 = costLS1 + costR1
                
                if cost1 > 2 * cost0 or np.isnan(cost1):
                    t = t / 2
                    x1 = x10.copy()
                    x2 = x20.copy()
                else:
                    break
            return x1, x2
            
        @staticmethod
        def optimize_k(x1, x2, k, y1, y2, tmpmu, params: MinRankKernelParam):
            ep = 1e-3
            i = 0
            k0 = k.copy()
            gradLS1 = 2 * fftconvolve(np.rot90(x1, 2), fftconvolve(x1, k, 'valid'), 'valid')
            gradLS2 = 2 * fftconvolve(np.rot90(x2, 2), fftconvolve(x2, k, 'valid'), 'valid')
            Ak = gradLS1 + gradLS2 + 2 * tmpmu * k

            b = 2 * fftconvolve(np.rot90(x1, 2), y1, 'valid') + 2 * fftconvolve(np.rot90(x2, 2), y2, 'valid')
            b = b + 2 * tmpmu * k0
            r = b - Ak

            d = r.copy()
            test = np.concatenate(r)
            e1 = np.matmul(np.transpose(np.concatenate(r)), np.concatenate(r))
            e0 = e1

            while e1 > ep * e0 and i < params.kmax:
                gradLS1 = 2 * fftconvolve(np.rot90(x1, 2), fftconvolve(x1, d, 'valid'), 'valid')
                gradLS2 = 2 * fftconvolve(np.rot90(x2, 2), fftconvolve(x2, d, 'valid'), 'valid')
                Ad = gradLS1 + gradLS2 + 2 * tmpmu * d; 
                alpha = e1 / (np.matmul(np.transpose(np.concatenate(d)), np.concatenate(Ad)))
                k = k + alpha * d
                
                if not np.mod(i, 50):
                    
                    gradLS1 = 2 * fftconvolve(np.rot90(x1, 2), fftconvolve(x1, k, 'valid') - y1, 'valid')
                    gradLS2 = 2 * fftconvolve(np.rot90(x2, 2), fftconvolve(x2, k, 'valid') - y2, 'valid')
                    gradLS = gradLS1 + gradLS2 + 2 * tmpmu * (k - k0)

                    r = -gradLS
                else:
                    r = r - alpha * Ad
                
                e0 = e1
                e1 = np.matmul(np.transpose(np.concatenate(r)), np.concatenate(r))
                beta = e1 / e0
                d = r + beta * d
                i = i + 1
            
            return k

        @staticmethod
        def optimize_rank(k0, tau, params: MinRankKernelParam):
            mu = 1
            Xh = k0.copy()
            w = mu * np.ones((Xh.shape[0],1))

            for _ in range(params.rmax):
                Uh, Sh ,Vh = np.linalg.svd(Xh)
                Lh = np.dot(Uh * np.maximum(Sh - (tau * np.diag(w)), np.zeros(Sh.shape)), Vh) #
                SLh = np.linalg.svdvals(Lh)
                w = np.ones((w.shape[0],1)) / (SLh+params.delta) 

            return Lh
