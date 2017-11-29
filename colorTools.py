import numpy as np
import cv2
import laneFindingPipeline


def circleKernel(ksize):
    kernel = cv2.getGaussianKernel(ksize, 0)
    kernel = (kernel * kernel.T > kernel.min()/3).astype('uint8')
    return kernel


def equalizeHist(img):
    img = np.copy(img)
    for i in range(3):
        img[:, :, i] = cv2.equalizeHist(img[:, :, i])
    return img


def morphologicalSmoothing(img, ksize=10):
    # For binary images only.
    # Circular kernel:
    kernel = cv2.getGaussianKernel(ksize, 0)
    kernel * kernel.T > kernel.min() / 3
    # Close holes:
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Despeckle:
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


def uint8scale(vec, lo=0):
    vec = np.copy(vec)
    vec -= vec.min()
    if vec.max() != 0:
        vec /= vec.max()
    vec *= (255 - lo)
    vec += lo
    return vec.astype('uint8')


dilate = lambda image, ksize=5, iterations=1: cv2.dilate(image.astype('uint8'), circleKernel(ksize), iterations=iterations)


erode  = lambda image, ksize=5, iterations=1: cv2.erode( image.astype('uint8'), circleKernel(ksize), iterations=iterations)


opening= lambda image, ksize=5, iterations=1: cv2.morphologyEx(
    image.astype('uint8'), cv2.MORPH_OPEN, np.ones((ksize,ksize)), iterations=iterations
)


blur = lambda image, ksize=5: cv2.GaussianBlur(image, (ksize, ksize), 0)


class CountSeekingThreshold:
    
    def __init__(self, initialThreshold=150):
        self.threshold = initialThreshold
        self.iterationCounts = []
        
    def __call__(self, channel, goalCount=10000, countTol=200):
        
        def getCount(threshold):
            mask = channel > np.ceil(threshold)
            return mask, mask.sum()
        
        threshold = self.threshold
        
        under = 0
        over = 255
        getThreshold = lambda : (over - under) / 2 + under
        niter = 0
        while True:
            mask, count = getCount(threshold)
            if (
                abs(count - goalCount) < countTol
                or over - under <= 1
            ):
                break

            if count > goalCount:
                # Too many pixels got in; threshold needs to be higher.
                under = threshold
                threshold = getThreshold()
            else: # count < goalCout
                if threshold > 254 and getCount(254)[1] > goalCount:
                    # In the special case that opening any at all is too bright, die early.
                    threshold = 255
                    mask = np.zeros_like(channel, 'bool')
                    break
                over = threshold
                threshold = getThreshold()
            niter += 1
                
        out =  max(min(int(np.ceil(threshold)), 255), 0)
        self.threshold = out
        self.iterationCounts.append(niter)
        return mask, out
