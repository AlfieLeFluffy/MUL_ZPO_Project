import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, fftconvolve
import cProfile, pstats, io
from pstats import SortKey
from src.image import CImage

class Deconvolve:

    #def conv2_center(_image, _kernel, _shape):
    #    return np.fft.irfft2(np.fft.rfft2(_image, _shape) * np.fft.rfft2(_kernel, _shape), _shape)

    class MinRankKernel:

        class Param:
            def __init__(self):
                self.tx = 1e-2
                self.tau = 1e-5#3e-4#1e-5

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
                self.sigma = True

                # image reg param
                self.lamb = 80

                # kernel threshold (threshold * max)
                self.threshold = 0.05
                self.mu = 1

                # k step total iter #
                self.iterkrank = 10

                # whether show inter results (suggested true to tune...)
                self.verbose = True

        @staticmethod
        def deconvolve_cry(_image, _ksize, params: Param):
            assert (np.mod(_ksize,2)==1), "Kernel size must be odd!"

            plt.figure()
            plt.imshow(_image)
            plt.plot()

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

                #print(kernel)
                #if i == 1:
                #    exit(0)

                y1 = fftconvolve(smally, dx, "valid")
                y2 = fftconvolve(smally, dy, "valid")

                x1, x2, kernel = Deconvolve.MinRankKernel.blind_deconvolve(y1, y2, x1, x2, kernel, (params.tau * ((i+1) / len(scale))), L2norm, params)
                y1, x1, kernel = Deconvolve.MinRankKernel.center_kernel_separate(y1, x1, kernel)
                y2, x2, kernel = Deconvolve.MinRankKernel.center_kernel_separate(y2, x2, kernel)
                
                if True:
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

            kernel[kernel < np.max(kernel) * params.threshold] = 0
            kernel = kernel / np.sum(kernel)
        
            return x1, kernel
        
        @staticmethod
        def center_kernel_separate(y, x, k):
            k_shape = k.shape
            mu_y = np.sum((range(1,k_shape[0]+1) * np.sum(k,1).T))
            mu_x = np.sum((range(1,k_shape[1]+1) * np.sum(k,0)))
            
            offset_x = int(np.round( np.floor(k.shape[1] / 2) + 1 - mu_x ))
            offset_y = int(np.round( np.floor(k.shape[0] / 2) + 1 - mu_y ))

            print(f'CenterKernel: weightedMean[{mu_x-1},{mu_y-1}] offset[{offset_x},{offset_y}]')

            shift_kernel = np.zeros((np.abs(offset_y * 2) + 1, np.abs(offset_x * 2) + 1))
            shift_kernel[np.abs(offset_y) + offset_y, np.abs(offset_x) + offset_x] = 1
            
            kshift = fftconvolve(k, shift_kernel, "same")
            xshift = fftconvolve(x, np.flip(np.flip(shift_kernel,0),1), "same")
            yshift = fftconvolve(y, np.flip(np.flip(shift_kernel,0),1), "same")

            return yshift, xshift, kshift

        @staticmethod
        def blind_deconvolve(y1, y2, x1, x2, kernel, tau, L2norm, params: Param):
            if np.sum(np.abs(x1)) == 0:
                x1 = y1
            if np.sum(np.abs(x2)) == 0:
                x2 = y2

            x1 = x1 * L2norm  / Deconvolve.norm2(x1)
            x2 = x2 * L2norm  / Deconvolve.norm2(x2)

            ksz = kernel.shape[0]
            bhs = int(np.floor(ksz / 2))
            y1v = y1[bhs : y1.shape[0] - bhs, bhs : y1.shape[1] - bhs]
            y2v = y2[bhs : y2.shape[0] - bhs, bhs : y2.shape[1] - bhs]

            #changes = np.zeros((params.imax, 1))
            #k0 = kernel.copy()
            for i in range(params.imax):
                print(f'Iteration {i+1}...', flush=True)
                x1, x2 = Deconvolve.MinRankKernel.optimize_x(x1, x2, kernel, y1, y2, params)
                for iter in range(params.iterkrank):
                    if iter == 0:
                        tmpmu = 0
                    else:
                        tmpmu = params.mu * np.exp(iter) / np.exp(params.iterkrank)
                    
                    kernel = Deconvolve.MinRankKernel.optimize_k(x1, x2, kernel, y1v, y2v, tmpmu, params)

                    if params.sigma:
                        kernel = Deconvolve.MinRankKernel.optimize_rank(kernel, tau, params)
                    
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
        def optimize_x(_x1, _x2, _k, _y1, _y2, params: Param):
            x1 = _x1 * 6 / Deconvolve.norm2(_x1)
            x2 = _x2 * 6 / Deconvolve.norm2(_x2)

            tmp1 = fftconvolve(x1, _k, 'same') - _y1
            tmp2 = fftconvolve(x2, _k, 'same') - _y2
            costLS0 = params.lamb / 2 * (np.matmul(np.concatenate(tmp1).T, np.concatenate(tmp1)) + np.matmul(np.concatenate(tmp2).T, np.concatenate(tmp2)))
            costR0 =  Deconvolve.norm1(x1) / Deconvolve.norm2(x1) + Deconvolve.norm1(x2) / Deconvolve.norm2(x2)
            cost0 = costLS0 + costR0

            t = params.tx
            tp = 1e-4

            while t > tp:
                x10 = x1.copy()
                x20 = x2.copy()
                for _ in range(params.ximax):
                    l21 = Deconvolve.norm2(x1)
                    l22 = Deconvolve.norm2(x2)
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
                costR1 = Deconvolve.norm1(x1) / Deconvolve.norm2(x1) + Deconvolve.norm1(x2) / Deconvolve.norm2(x2)
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
        def optimize_rank(k0, tau, params: Param):
            mu = 1
            Xh = k0.copy()
            w = mu * np.ones((Xh.shape[0],1))

            for _ in range(params.rmax):
                Uh, Sh ,Vh = np.linalg.svd(Xh)
                Lh = np.dot(Uh * np.maximum(Sh - (tau * np.diag(w)), np.zeros(Sh.shape)), Vh) #
                SLh = np.linalg.svdvals(Lh)
                w = np.ones((w.shape[0],1)) / (SLh+params.delta) 

            return Lh

    class BregmanDeconvolution:
        def __init__(self):
            self.lookup_v = 0
            self.xx = 0
            self.known_beta = np.array([])
            self.known_alpha = np.array([])

        def bregman_deconvolution(self, _image, _kernel, _lambda, _alpha):
            beta = 400
            initer_max = 1
            outiter_max = 50

            m = _image.shape[0]
            n = _image.shape[1]
            km = _kernel.shape[0]
            kn = _kernel.shape[1]
            ks = int(np.floor(_kernel.shape[0]/2))
            g = _image.copy()

            if (np.mod(km,2)!=1) or (np.mod(kn,2)!=1):
                raise Exception("Kernel must be odd!")
            

            dx = np.array([[1, -1]])
            dy = np.array([[1],[-1]]) #np.transpose(dx)
            dxt = np.array([[-1, 1]]) #np.flip(dx,0)
            dyt = np.array([[-1],[1]]) #np.flip(dy,1)

            Ktf, KtK, DtD, Fdx, Fdy = self.compute_constants(_image, _kernel, dx, dy)
            
            gx = fftconvolve(g, dx, 'valid')
            gy = fftconvolve(g, dy, 'valid')

            #fx = convolve2d(_image, dx, 'valid')
            #fy = convolve2d(_image, dy, 'valid')

            #ks = _kernel.shape[0]
            #ks2 = int(np.floor(ks / 2))

            # store some of the statistics
            #lcost = np.array([])
            #pcost = np.array([])
            outiter = 0

            bx = np.zeros(gx.shape)
            by = np.zeros(gy.shape)
            wx = gx.copy()
            wy = gy.copy()

            totiter = 1
            #gk = convolve2d(g, _kernel, 'same')
                
            #lcost.insert(totiter, (_lambda / 2) * np.pow(ImageProcessor.Deconvolve.norm2(np.concatenate(gk) - np.concatenate(_image)),2))
            #pcost.insert(totiter, np.sum((np.pow(np.abs(np.concatenate(gx)), _alpha))))
            #pcost.insert(totiter, pcost[totiter] + np.sum((np.pow(np.abs(np.concatenate(gy), _alpha)))))


            for outiter in range(outiter_max):
                print(f'Outer iteration {outiter+1}')

                for initer in range(initer_max):
                    totiter += 1
                    
                    if _alpha == 1:
                        tmpx = beta * (gx + bx)
                        betax = beta
                        tmpx = tmpx / betax
                        
                        tmpy = beta * (gy + by)
                        betay = beta
                        tmpy = tmpy / betay
                        betay = betay
                        wx = np.multiply(np.maximum(np.abs(tmpx) - np.ones(tmpx.shape)/ betax, np.zeros(tmpx.shape)), np.sign(tmpx))
                        wy = np.multiply(np.maximum(np.abs(tmpy) - np.ones(tmpy.shape)/ betay, np.zeros(tmpy.shape)), np.sign(tmpy))
                    else:
                        wx = self.solve_image_bregman(gx + bx, beta, _alpha)
                        wy = self.solve_image_bregman(gy + by, beta, _alpha)
                        
                    bx = bx - wx + gx
                    by = by - wy + gy
                    
                    wx1 = fftconvolve(wx - bx, dxt, 'full')
                    wy1 = fftconvolve(wy - by, dyt, 'full')
                    #tmp = np.zeros(g.shape)
                    
                    #gprev = g.copy()
                    #gxprev = gx.copy()
                    #gyprev = gy.copy()
                    
                    num = _lambda * Ktf + beta * np.fft.fft2(wx1 + wy1)
                    denom = _lambda * KtK + beta * DtD
                    Fg = np.divide(num, denom)
                    g = np.real(np.fft.ifft2(Fg))
                    
                    gx = fftconvolve(g, dx, 'valid')
                    gy = fftconvolve(g, dy, 'valid')
                    #gk = convolve2d(g, _kernel, 'same')
                    #lcost.insert(totiter, (_lambda / 2) * np.pow(ImageProcessor.Deconvolve.norm2(np.concatenate(gk) - np.concatenate(_image)),2))
                    #pcost.insert(totiter, np.sum((np.pow(np.abs(np.concatenate(gx)), _alpha))))
                    #pcost.insert(totiter, pcost[totiter] + np.sum((np.pow(np.abs(np.concatenate(gy), _alpha)))))

            return g
        
        def compute_constants(self, _image, _kernel, _dx, _dy):
            sizef = _image.shape
            otfk  = Deconvolve.psf2otf(_kernel, sizef)
            Ktf = np.conj(otfk) * np.fft.fft2(_image)
            KtK = np.pow(np.abs(otfk), 2)
            Fdx = np.pow(np.abs(Deconvolve.psf2otf(_dx, sizef)),2)
            Fdy = np.pow(np.abs(Deconvolve.psf2otf(_dy, sizef)),2)
            DtD = Fdx + Fdy
            return Ktf, KtK, DtD, Fdx, Fdy
        
        def solve_image_bregman(self, _input, _beta, _alpha):
            rang = 10
            step  = 0.0001

            ind = np.array([np.where(self.known_beta==_beta), np.where(self.known_alpha==_alpha)])

            if self.known_alpha.size == 0 or self.known_beta.size == 0:
                self.xx = np.array(range(-rang, rang, step))
            
            if np.any(ind):
                print(f'Reusing lookup table for beta {_beta} and alpha {_alpha}')
                # already computed 
                w = np.interp1(np.transpose(self.xx), np.transpose(self.lookup_v[ind,:]), np.concatenate(_input), 'linear', 'extrap')
                w = np.reshape(w, _input.shape)
            else:
                # now go and recompute xx for new value of beta and alpha
                tmp = self.compute_w(self.xx ,_beta, _alpha)
        
        def compute_w(self, _input, _beta, _alpha):
            if (np.abs(_alpha - 1) < 1e-9):
                # assume alpha = 1.0
                w = self.compute_w1(_input, _beta)
                return w

            if (np.abs(_alpha - 2/3) < 1e-9):
                # assume alpha = 2/3
                w = self.compute_w23(_input, _beta)
                return w

            if (abs(_alpha - 1/2) < 1e-9):
                # assume alpha = 1/2
                w = self.compute_w12(_input, _beta)
                return w
        
        def compute_w1(self , _input, _beta):
            # solve a simple max problem for alpha = 1
            w = np.maximum(np.abs(_input) - 1/_beta, np.zeros(_input.shape)) * np.sign(_input)
            return w
        
        def compute_w12(self , _input, _beta):
            # solve a cubic equation
            # for alpha = 1/2

            epsilon = 1e-6
            k = -0.25 / np.pow(_beta,2)
            m = np.ones(_input.shape) * k * np.sign(_input)

            t1 = (2/3) * _input
            v2 = _input * _input
            v3 = v2 * _input
            t2 = np.exp(np.log(-27 * m - 2 * v3 + (3 * np.sqrt(3)) * np.sqrt(27 * np.pow(m,2) + 4 * m * v3))/3)
            t3 = v2/t2


    @staticmethod
    def deconvlove_image(_image: CImage, _ksize, _params, _finalization=True):
        nb_lambda = 3000
        nb_alpha = 0.5
        use_ycbcr = True
        print(_image.type)

        if use_ycbcr:
            if len(_image.size) == 3:
                if _image.size[2] == 3:
                    ycbcr: CImage = _image.rgb2ycbcr()
                else:
                    raise Exception("What is this picture?")
            else:
                ycbcr = _image
            nb_alpha = 1
        
        pr = cProfile.Profile()
        pr.enable()
        _, kernel = Deconvolve.MinRankKernel.deconvolve_cry(ycbcr.data[:,:,0], _ksize, _params)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open("temp.txt", "w") as f:
            f.write(s.getvalue())
        #print(s.getvalue())

        #kernel = cv2.getGaussianKernel(9, 1) @ cv2.getGaussianKernel(9, 1).T
        bhs = int(np.floor(kernel.shape[0] / 2))
        
        if _finalization:
            ypad = np.pad(ycbcr.data[:,:,0], ((bhs,bhs),(bhs,bhs)), "edge")
            ypadtemp = Deconvolve.edgetaper(ypad, kernel)
            for _ in range(2):
                ypadtemp = Deconvolve.edgetaper(ypadtemp, kernel)

            bregmain_decon = Deconvolve.BregmanDeconvolution()
            image = bregmain_decon.bregman_deconvolution(ypadtemp, kernel, nb_lambda, nb_alpha)

            plt.figure()
            plt.subplot(231)
            plt.title("_image")
            plt.imshow(_image.data)
            plt.subplot(232)
            plt.title("ycbcr")
            plt.imshow(ycbcr.data)
            plt.subplot(234)
            plt.title("ypad")
            plt.imshow(ypad)
            plt.subplot(235)
            plt.title("ypadtemp")
            plt.imshow(ypadtemp)
            plt.subplot(236)
            plt.title("image")
            plt.imshow(image)
            plt.show()
        else:
            image = ycbcr

        return image, kernel
    
    
    
    @staticmethod
    def otf2psf(otf, shape=None):
        if shape is None:
            shape = otf.shape
        
        # Inverse Fourier transform
        psf = np.fft.ifftn(otf, shape)
        
        # Shift the zero-frequency component to the center
        psf = np.fft.fftshift(psf)
        
        # Take the real part, as the PSF should be real-valued
        psf = np.real(psf)
        
        # Normalize the PSF so that its sum is 1
        psf /= np.sum(psf)
        
        return psf

    @staticmethod
    def zero_pad(image, shape, position='corner'):
        """
        Extends image to a certain size with zeros

        Parameters
        ----------
        image: real 2d `numpy.ndarray`
            Input image
        shape: tuple of int
            Desired output shape of the image
        position : str, optional
            The position of the input image in the output one:
                * 'corner'
                    top-left corner (default)
                * 'center'
                    centered

        Returns
        -------
        padded_img: real `numpy.ndarray`
            The zero-padded image

        """
        shape = np.asarray(shape, dtype=int)
        imshape = np.asarray(image.shape, dtype=int)

        if np.all(imshape == shape):
            return image

        if np.any(shape <= 0):
            raise ValueError("ZERO_PAD: null or negative shape given")

        dshape = shape - imshape
        if np.any(dshape < 0):
            raise ValueError("ZERO_PAD: target size smaller than source one")

        pad_img = np.zeros(shape, dtype=image.dtype)

        idx, idy = np.indices(imshape)

        if position == 'center':
            if np.any(dshape % 2 != 0):
                raise ValueError("ZERO_PAD: source and target shapes "
                                "have different parity.")
            offx, offy = dshape // 2
        else:
            offx, offy = (0, 0)

        pad_img[idx + offx, idy + offy] = image

        return pad_img

    @staticmethod
    def psf2otf(psf, shape):
        """
        Convert point-spread function to optical transfer function.

        Compute the Fast Fourier Transform (FFT) of the point-spread
        function (PSF) array and creates the optical transfer function (OTF)
        array that is not influenced by the PSF off-centering.
        By default, the OTF array is the same size as the PSF array.

        To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
        post-pads the PSF array (down or to the right) with zeros to match
        dimensions specified in OUTSIZE, then circularly shifts the values of
        the PSF array up (or to the left) until the central pixel reaches (1,1)
        position.

        Parameters
        ----------
        psf : `numpy.ndarray`
            PSF array
        shape : int
            Output shape of the OTF array

        Returns
        -------
        otf : `numpy.ndarray`
            OTF array

        Notes
        -----
        Adapted from MATLAB psf2otf function
        Arrays of higher dimension that 2D are also supported

        """
        if np.all(psf == 0):
            return np.zeros_like(psf)

        inshape = psf.shape
        # Pad the PSF to outsize
        psf = Deconvolve.zero_pad(psf, shape, position='corner')

        # Circularly shift OTF so that the 'center' of the PSF is
        # [0,0] element of the array
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)

        # Compute the OTF
        otf = np.fft.fftn(psf)

        # Estimate the rough number of operations involved in the FFT
        # and discard the PSF imaginary part if within roundoff error
        # roundoff error  = machine epsilon = sys.float_info.epsilon
        # or np.finfo().eps
        n_ops = np.sum(psf.size * np.log2(psf.shape))
        otf = np.real_if_close(otf, tol=n_ops)

        return otf

    @staticmethod
    def edgetaper(img, psf):
        """
        Taper edges of an image to reduce DFT ringing artifacts.
        """
        h, w = img.shape
        
        # 1. Create a tapering mask (alpha) based on the PSF
        # A simple approach is a border that fades out
        alpha = np.ones((h, w), dtype=np.float32)
        
        # Create a margin based on PSF size
        margin_y = int(psf.shape[0] * 2)
        margin_x = int(psf.shape[1] * 2)
        
        # Apply a gentle fade (linear or gaussian) at the borders
        for i in range(min(margin_y, h//2)):
            alpha[i, :] *= (i / margin_y)
            alpha[h-1-i, :] *= (i / margin_y)
        for i in range(min(margin_x, w//2)):
            alpha[:, i] *= (i / margin_x)
            alpha[:, w-1-i] *= (i / margin_x)
        
        # Smooth the mask to avoid sharp, artificial transitions
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        # 2. Blur the original image to create a "background" for the edges
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        
        # 3. Blend the original and blurred image using the mask
        # Need to handle color channels
        if len(img.shape) == 3:
            alpha = cv2.merge([alpha, alpha, alpha])
            
        tapered_img = alpha * img + (1.0 - alpha) * blurred
        
        return tapered_img.astype(np.uint8)

    @staticmethod
    def norm2(x):
        return np.linalg.norm(np.concatenate(x).T)
        #return np.sqrt(np.sum(np.pow(np.abs(x),2)))

    @staticmethod
    def norm1(x):
        return np.linalg.norm(np.concatenate(x).T, 1)
        #return np.sum(np.abs(x))