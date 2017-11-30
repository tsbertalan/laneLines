from cvflow.baseOps import *

class CountSeekingThresholdOp(Boolean):
    
    def __init__(self, parent, initialThreshold=150, goalCount=10000, countTol=200):
        super().__init__()
        assert isinstance(parent, Mono)
        self.addParent(parent)
        self.threshold = initialThreshold
        self.goalCount = goalCount
        self.countTol = countTol
        self.iterationCounts = []
        
    @cached
    def value(self):
        channel = self.parent()
        goalCount = self.goalCount
        countTol = self.countTol
        
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
        return mask

    def __str__(self):
        return 'Count=%d threshold (tol %s; current %s).' % (self.goalCount, self.countTol, self.threshold)



class DilateSobel(Boolean):

    def __init__(self, singleChannel, postdilate=True, preblurksize=13, sx_thresh=20, dilate_kernel=(2, 4), dilationIterations=3):
        super().__init__()
        self.sx_thresh = sx_thresh
        self.dilate_kernel = dilate_kernel
        self.dilationIterations = dilationIterations

        # Add a little *more* blurring.
        self.blur = Blur(singleChannel, ksize=preblurksize)
        self.sobelx = Sobel(self.blur, xy='x')

        # Sobel mask.
        # mask_neg = AsBoolean(AsType(LessThan(   self.sobelx, -sx_thresh), 'float32'))
        # mask_pos = AsBoolean(AsType(GreaterThan(self.sobelx,  sx_thresh), 'float32'))
        self.mask_neg = LessThan(   self.sobelx, -sx_thresh)
        self.mask_pos = GreaterThan(self.sobelx,  sx_thresh)

        kernel_midpoint = dilate_kernel[1] // 2

        # Dilate mask to the left.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, 0:kernel_midpoint] = 0
        self.dmask_neg = GreaterThan(Dilate(self.mask_neg, kernel, iterations=dilationIterations), 0.)

        # Dilate mask to the right.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, kernel_midpoint:] = 0
        self.dmask_pos = GreaterThan(Dilate(self.mask_pos, kernel, iterations=dilationIterations), 0.)

        # self.sxbinary = AsBoolean(AsType(And(self.dmask_pos, self.dmask_neg), 'uint8'))
        self.sxbinary = AsBoolean(And(self.dmask_pos, self.dmask_neg))

        if postdilate:
            self.sxbinary = Dilate(self.sxbinary)

        self.addParent(self.sxbinary)

        self.members = [
            self.sobelx, self.mask_neg, self.mask_pos, self.dmask_neg, 
            self.dmask_pos, self.sxbinary, self.blur, self
        ]
        for m in self.members:
            if (
                m not in [self.blur]
                and len(m.parents) > 0
            ):
                self.members.append(m.parents[0])

    @cached
    def value(self):
        return self.sxbinary()


class SobelClip(Op):

    def __init__(self, channel, threshold=None):
        super().__init__()

        # Adaptive thresholding of color.
        if threshold is None:
            threshold = CountSeekingThresholdOp(channel)
        self.threshold = threshold

        # Dilated masks of the threshold.
        self.narrow = Dilate(threshold, kernel=10, iterations=5)
        self.wide = Dilate(self.narrow, kernel=10, iterations=5)
        
        # Restricted Sobel-X
        self.toSobel = And(channel, Not(self.wide))

        self.sobel = DilateSobel(self.toSobel)
        self.clippedSobel = And(self.sobel, self.narrow)
        self.addParent(self.clippedSobel)

        self.members =  [self.threshold, self.narrow, self.wide, self.toSobel, self.sobel, self.clippedSobel, self]

    @cached
    def value(self):
        return self.clippedSobel()

