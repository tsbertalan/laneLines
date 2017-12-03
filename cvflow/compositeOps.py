import graphviz
from cvflow.baseOps import *
from cvflow.workers import *
from cvflow.misc import cached
from cvflow.multistep import MultistepOp


class DilateSobel(MultistepOp, Boolean):

    def __init__(self, singleChannelOrPreviousDilateSobel, **kwargs):
        defaults = dict(
            postdilate=True, preblurksize=13, sx_thresh=20, 
            dilate_kernel=(2, 4), dilationIterations=3, reversed=False,
        )
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.defaults = defaults

        # To get both trenches and ridges,
        # we might reuse some stuff from a DilateSobel.
        if isinstance(singleChannelOrPreviousDilateSobel, DilateSobel):
            self.input = singleChannelOrPreviousDilateSobel.input.parent()
            # If we're reusing some stuff, it's because we want to compute
            # this filter again, but for the inverted features.
            self.reversed = not singleChannelOrPreviousDilateSobel.reversed
            self.mask_neg = singleChannelOrPreviousDilateSobel.mask_neg
            self.mask_pos = singleChannelOrPreviousDilateSobel.mask_pos
        else:
            self.input = singleChannelOrPreviousDilateSobel
            # If not, we need to do these computations here.
            self.assertProp(singleChannelOrPreviousDilateSobel, isMono=True)

            # Add a little *more* blurring.
            blur = Blur(self.input, ksize=self.preblurksize)
            sobelx = Sobel(blur, xy='x')

            # Sobel mask.
            self.mask_neg = sobelx < -self.sx_thresh
            self.mask_pos = sobelx >  self.sx_thresh
            self.includeInMultistep([
                blur, sobelx, self.mask_neg, self.mask_pos
            ])

        # Define one-sided kernels.
        dilate_kernel = self.dilate_kernel
        kernel_midpoint = dilate_kernel[1] // 2
        rightdilate = np.ones(dilate_kernel, 'uint8')
        rightdilate[:, kernel_midpoint:] = 0
        leftdilate = np.ones(dilate_kernel, 'uint8')
        leftdilate[:, 0:kernel_midpoint] = 0

        if self.reversed:
            # To find dark lines, we look for down followed by up.
            # So, negative slopes are dilated to the right.
            kernelNeg = rightdilate
            kernelPos = leftdilate
        else:
            kernelNeg = leftdilate
            kernelPos = rightdilate

        # Dilate mask to the left.
        dmask_neg = Dilate(self.mask_neg, kernelNeg, iterations=self.dilationIterations)

        # Dilate mask to the right.
        dmask_pos = Dilate(self.mask_pos, kernelPos, iterations=self.dilationIterations)

        # Look for intersections of the left and right masks.
        sxbinary = dmask_pos & dmask_neg
        sxbinary.isBinary = True

        if self.postdilate:
            sxbinary = Dilate(sxbinary)

        self.output = sxbinary

        self.includeInMultistep([
            self.mask_neg, self.mask_pos, dmask_pos, dmask_neg, 
            sxbinary, sxbinary.parent().parent(), 
        ])

        # For analysis, also allow for outputting a multicolored dilation.
        zero = sxbinary*0; zero.hidden = True
        self.constructColorOutpout(zero, dmask_pos, dmask_neg).nodeName = 'dmask (%s)' % self
        self.constructColorOutpout(zero, self.mask_pos, self.mask_neg).nodeName = 'mask (%s)' % self
        super().__init__()

    @stringFallback
    def __str__(self):
        return 'DilateSobel(%s)' % ', '.join([
            '%s=%s' % (k, getattr(self, k)) 
            for (k, v) in self.defaults.items()
            if getattr(self, k) != v
        ])


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

class LightPatchRemover(MultistepOp, Boolean):

    def __init__(self, channel, bigK=128):
        self.assertProp(channel, isMono=True)
        self.input = channel

        # Box-average the image--convolve it with a really
        # big square filter. A square filter works better
        # than a Gaussian blur at ensuring no zero values 
        # (which helps with the consistency of the next part).
        localAverage = Convolve(self.input, kernel=bigK)

        # We next divide pixels by their local average.
        # If the orignal signal looks like this
        # (as at the border of a light or shadow patch):
        # 255``````|
        #          |
        # 16       |_____
        # The local box-average will look like this:
        # 255`````\
        #          \
        # 16        \____
        # Far from the boundary, these divide to one, but,
        # as we pass the shock from left to right, the quotient
        # first goes up, then down.
        # 2      /\  
        # 1-----/  \  /----
        # 0         \/
        nearOne = self.input // localAverage

        # Finally, we shift this down to near zero,take the absolute value,
        # and cast the result back to uint8.
        self.output = deEmphasized = AsType(Abs((v // convv) - 1), 'uint8', scaleUintTo255=True)

        self.includeInMultistep([
            localAVerage, nearOne, deEmphasized,
        ])
        super().__init__()

        # We then realize that we have just implemented a
        # worse version of an XY Sobel filter.
