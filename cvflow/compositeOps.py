import graphviz
from cvflow.baseOps import *
from cvflow.workers import *
from cvflow.misc import cached
from cvflow.multistep import MultistepOp


class DilateSobel(MultistepOp, Boolean):

    def __init__(self, singleChannel, 
        postdilate=True, preblurksize=13, sx_thresh=20, 
        dilate_kernel=(2, 4), dilationIterations=3
        ):
        self.sx_thresh = sx_thresh
        self.dilate_kernel = dilate_kernel
        self.dilationIterations = dilationIterations
        self.assertProp(singleChannel, isMono=True)
        self.input = singleChannel

        # Add a little *more* blurring.
        blur = Blur(self.input, ksize=preblurksize)
        sobelx = Sobel(blur, xy='x')

        # Sobel mask.
        mask_neg = sobelx < -sx_thresh
        mask_pos = sobelx >  sx_thresh

        kernel_midpoint = dilate_kernel[1] // 2

        # Dilate mask to the left.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, 0:kernel_midpoint] = 0
        dmask_neg = Dilate(mask_neg, kernel, iterations=dilationIterations)

        # Dilate mask to the right.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, kernel_midpoint:] = 0
        dmask_pos = Dilate(mask_pos, kernel, iterations=dilationIterations)

        # Look for intersections of the left and right masks.
        sxbinary = dmask_pos & dmask_neg
        sxbinary.isBinary = True

        if postdilate:
            sxbinary = Dilate(sxbinary)

        self.output = sxbinary

        self.includeInMultistep([
            blur, mask_neg, mask_pos, dmask_pos, dmask_neg, sxbinary, sxbinary.parent().parent()
        ])
        super().__init__()


class SobelClip(MultistepOp, Boolean):

    def __init__(self, channel, 
        narrowIterations=5, wideIterations=5,
        narrowKernel=10, wideKernel=10,
        thresholdKwargs={},
        **dilateSobelKwargs
        ):

        self.assertProp(channel, isMono=True)
        self.input = channel

        # Adaptive thresholding of color.
        threshold = CountSeekingThreshold(self.input, **thresholdKwargs)

        # Dilated masks of the threshold.
        narrow = Dilate(threshold, kernel=narrowKernel, iterations=narrowIterations)
        wide = Dilate(narrow, kernel=wideKernel, iterations=wideIterations)
        
        # Restricted Sobel-X
        toSobel = self.input & wide
        toSobel.isMono = True

        sobel = DilateSobel(toSobel, **dilateSobelKwargs)
        clippedSobel = sobel & narrow
        
        self.output = clippedSobel

        self.includeInMultistep([
            sobel, threshold, narrow, wide, toSobel, clippedSobel
        ])
        super().__init__()
