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


opening = lambda image, ksize=5, iterations=1: cv2.morphologyEx(
    image.astype('uint8'), cv2.MORPH_OPEN, np.ones((ksize,ksize)), iterations=iterations
)


blur = lambda image, ksize=5: cv2.GaussianBlur(image, (ksize, ksize), 0)


# class CountSeekingThreshold:
    
#     def __init__(self, initialThreshold=150):
#         self.threshold = initialThreshold
#         self.iterationCounts = []
        
#     def __call__(self, channel, goalCount=10000, countTol=200):
        
#         def getCount(threshold):
#             mask = channel > np.ceil(threshold)
#             return mask, mask.sum()
        
#         threshold = self.threshold
        
#         under = 0
#         over = 255
#         getThreshold = lambda : (over - under) / 2 + under
#         niter = 0
#         while True:
#             mask, count = getCount(threshold)
#             if (
#                 abs(count - goalCount) < countTol
#                 or over - under <= 1
#             ):
#                 break

#             if count > goalCount:
#                 # Too many pixels got in; threshold needs to be higher.
#                 under = threshold
#                 threshold = getThreshold()
#             else: # count < goalCout
#                 if threshold > 254 and getCount(254)[1] > goalCount:
#                     # In the special case that opening any at all is too bright, die early.
#                     threshold = 255
#                     mask = np.zeros_like(channel, 'bool')
#                     break
#                 over = threshold
#                 threshold = getThreshold()
#             niter += 1
                
#         out =  max(min(int(np.ceil(threshold)), 255), 0)
#         self.threshold = out
#         self.iterationCounts.append(niter)
#         return mask, out

#     def sobelclip(self, channel, **kw):
#         # Adaptive thresholding of color.
#         tmask, thresh = self(channel, **kw)

#         # Dilated masks of the threshold.
#         narrow = ct.dilate(tmask, ksize=10, iterations=5)
#         wide = ct.dilate(narrow, ksize=10, iterations=5)
        
#         # Restricted Sobel-X
#         toSobel = np.copy(channel)
#         toSobel[np.logical_not(wide)] = 0
        
#         sobel = colorFilter.dilateSobel(toSobel)
#         clippedSobel = sobel & narrow

#         return tmask, thresh, narrow, wide, toSobel, sobel, clippedSobel


def _cached(method):
    def wrappedMethod(self, *args, **kwargs):
        if not hasattr(self, '__cache__'):
            self.__cache__ = {}
        key = '%s(*%s, **%s)' % (method.__name__, args, kwargs)
        if key in self.__cache__:
            out = self.__cache__[key]
        else:
            out = method(self, *args, **kwargs)
            self.__cache__[key] = out
        return out
    return wrappedMethod


def cached(method):
    return property(_cached(method))

def addId(__str__):
    def __newstr__(self):
        return '%s #%s' % (__str__(self), id(self))
    return __newstr__

class Op:

    def invalidateCache(self):
        self.__cache__ = {}
        for child in self.children:
            child.invalidateCache()

    @property
    def parents(self):
        if not hasattr(self, '_parents'):
            self._parents = []
        return self._parents

    @property
    def children(self):
        if not hasattr(self, '_children'):
            self._children = []
        return self._children

    def addChild(self, child):
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parnts.append(self)
        return child

    def addParent(self, parent):
        if not isinstance(parent, Op):
            parent = Constant(parent)
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)
        return parent

    @cached
    def value(self):
        raise NotImplementedError

    def __call__(self):
        return self.value

    def parent(self, index=0):
        return self.parents[index].value

    def assembleGraph(self, d=None, currentRecursionDepth=0):
        import networkx, graphviz
        if d is None:
            d = networkx.DiGraph()
        for child in self.children:
            if child not in d:
                d.add_edge(self, child)
                child.assembleGraph(d, currentRecursionDepth+1)
        return d

    def draw(self, ax=None):
        import networkx
        d = self.assembleGraph()
        if ax is not None:
            import matplotlib.pyplot as plt
            plt.sca(ax)
            networkx.draw_networkx(d)
            out = ax
        else:
            import graphviz
            from io import StringIO
            sio = StringIO()
            networkx.drawing.nx_pydot.write_dot(d, sio)
            dot = sio.getvalue()
            gv = graphviz.Source(dot)
            out = gv
        return out

    @addId
    def __str__(self):
        return type(self).__name__


class Constant(Op):

    def __init__(self, theConstant):
        self.theConstant = theConstant

    @property
    def value(self):
        return self.theConstant


class Lambda(Op):

    def __init__(self, f, *args):
        for arg in args:
            self.addParent(arg)
        self.f = f

    @cached
    def value(self):
        args = (arg.value for arg in self.parents)
        return self.f(*args)


class Mono(Op):
    
    def __init__(self, parent):
        self.addParent(parent)

    @property
    def value(self):
        return self.parent()


class Boolean(Mono):
    pass


class Not(Boolean):

    def __init__(self, parent):
        self.addParent(parent)

    @property
    def value(self):
        return np.logical_not(self.parent())


class Color(Op):
    
    def __init__(self, parent):
        self.addParent(parent)

    @cached
    def value(self):
        parent = self.parents[0]
        out = parent()
        if isinstance(parent, Mono):
            out = np.dstack((out, out, out))
        assert len(out.shape) == 3
        return out


class ColorSplit(Mono):

    def __init__(self, color, index):
        assert isinstance(color, Color)
        self.addParent(color)
        self.index = index

    @property
    def value(self):
        return self.parent()[:, :, self.index]


class BaseImage(Op):

    def __init__(self):
        pass

    @property
    def value(self):
        return self.image

    @value.setter
    def value(self, newimage):
        self.image = newimage
        if isinstance(self, Mono):
            assert len(newimage.shape) == 2
        else:
            assert len(newimage.shape) == 3
        self.invalidateCache()


class ColorImage(BaseImage, Color):
    pass


class MonoImage(BaseImage, Mono):
    pass


class Blur(Op):

    def __init__(self, parent, ksize=5):
        self.addParent(parent)
        assert ksize % 2
        self.ksize = ksize

    @cached
    def value(self):
        return cv2.GaussianBlur(self.parent(), (self.ksize, self.ksize), 0)

    @addId
    def __str__(self):
        return 'Blur(ksize=%d)' % self.ksize


class CircleKernel(Mono):

    def __init__(self, ksize, falloff=3):
        self.ksize = ksize
        self.falloff = falloff

    @cached
    def value(self):
        kernel = cv2.getGaussianKernel(self.ksize, 0)
        kernel = (kernel * kernel.T > kernel.min() / self.falloff).astype('uint8')
        return kernel


class Dilate(Mono):

    def __init__(self, parent, kernel=5, iterations=1):
        assert isinstance(parent, Mono)
        self.addParent(AsType(parent, 'uint8'))
        if isinstance(kernel, int):
            kernel = CircleKernel(kernel)
        self.addParent(kernel)
        self.iterations = iterations

    @cached
    def value(self):
        return cv2.dilate(
            self.parent(0), self.parent(1), iterations=self.iterations,
        )


class Erode(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        assert isinstance(parent, Mono)
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        self.iterations = iterations

    @cached
    def value(self):
        return cv2.erode(
            self.parent(0), self.parent(1), iterations=self.iterations
        )


class Opening(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        assert isinstance(parent, Mono)
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        self.iterations = iterations
        self.mono = True

    @cached
    def value(self):
        return cv2.morphologyEx(
            self.parent(0), self.parent(1), iterations=self.iterations
        )


class CountSeekingThresholdOp(Boolean):
    
    def __init__(self, parent, initialThreshold=150, goalCount=10000, countTol=200):
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


class And(Op):

    def __init__(self, parent1, parent2):
        p1m = isinstance(parent1, Mono)
        p2m = isinstance(parent2, Mono)
        assert p1m or p2m
        self.addParent(parent1)
        self.addParent(parent2)

    @cached
    def value(self):
        p1, p2 = self.parents
        if isinstance(p1, Mono) and isinstance(p2, Mono):
            out = p1() & p2()
        else:
            if isinstance(p1, Mono):
                assert isinstance(p2, Color)
                mono = p1
                color = p2
            else:
                assert isinstance(p1, Color)
                assert isinstance(p2, Mono)
                mono = p2
                color = p1
            out = np.copy(color())
            out[np.logical_not(mono())] = 0
        return out


class Sobel(Op):

    def __init__(self, channel, xy='x'):
        self.xy = xy
        self.addParent(channel)

    @cached
    def value(self):
        if self.xy == 'x':
            xy = (1, 0)
        elif self.xy == 'xy':
            xy = (1, 1)
        else:
            xy = (0, 1)
        return cv2.Sobel(self.parent(), cv2.CV_64F, *xy)


class _ElementwiseInequality(Boolean):

    def __init__(self, left, right, orEqualTo=False):
        self.addParent(left)
        self.addParent(right)
        self.orEqualTo = orEqualTo


class LessThan(_ElementwiseInequality):

    @cached
    def value(self):
        left, right = self.parents
        if self.orEqualTo:
            return left() <= right()
        else:
            return left() < right()


class GreaterThan(_ElementwiseInequality):

    @cached
    def value(self):
        left, right = self.parents
        if self.orEqualTo:
            return left() >= right()
        else:
            return left() > right()


class AsType(Op):

    def __init__(self, parent, kind):
        self.addParent(parent)
        self.kind = kind

    @cached
    def value(self):
        return self.parent().astype(self.kind)
        
    @addId
    def __str__(self):
        return 'AsType(%s)' % self.kind


class ScalarMultiply(Op):

    def __init__(self, parent, scalar):
        self.addParent(parent)
        self.scalar = scalar

    @cached
    def value(self):
        return self.parent() * self.scalar


class DilateSobel(Boolean):

    def __init__(self, singleChannel, postdilate=True, preblurksize=13, sx_thresh=20, dilate_kernel=(2, 4), dilationIterations=3):
        self.sx_thresh = sx_thresh
        self.dilate_kernel = dilate_kernel
        self.dilationIterations = dilationIterations

        # Add a little *more* blurring.
        blur = Blur(singleChannel, ksize=preblurksize)
        self.sobelx = Sobel(blur, xy='x')

        # Sobel mask.
        mask_neg = Boolean(AsType(LessThan(   self.sobelx, -sx_thresh), np.float32))
        mask_pos = Boolean(AsType(GreaterThan(self.sobelx,  sx_thresh), np.float32))

        kernel_midpoint = dilate_kernel[1] // 2

        # Dilate mask to the left.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, 0:kernel_midpoint] = 0
        self.dmask_neg = GreaterThan(Dilate(mask_neg, kernel, iterations=dilationIterations), 0.)

        # Dilate mask to the right.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, kernel_midpoint:] = 0
        self.dmask_pos = GreaterThan(Dilate(mask_pos, kernel, iterations=dilationIterations), 0.)

        self.sxbinary = Boolean(AsType(And(self.dmask_pos, self.dmask_neg), 'uint8'))

        if postdilate:
            self.sxbinary = Dilate(self.sxbinary)

        self.addParent(self.sxbinary)

    @cached
    def value(self):
        return self.sxbinary()


class SobelClip(Op):

    def __init__(self, channel, threshold=None):

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

    @cached
    def value(self):
        return self.clippedSobel()


pairFlags = {}
for name in dir(cv2):
    if name.startswith('COLOR_'):
        code = getattr(cv2, name)
        if code not in pairFlags:
            pairFlags[code] = name.upper()
        else:
            if len(name) < len(pairFlags[code]):
                pairFlags[code] = name.upper()

class CvtColor(Op):

    _pairFlagsCodes = pairFlags

    def __init__(self, image, pairFlag):
        self.addParent(image)
        self.pairFlag = pairFlag

    @property
    def value(self):
        return cv2.cvtColor(self.parent(), self.pairFlag)

    @addId
    def __str__(self):
        return 'CvtColor(%s)' % self._pairFlagsCodes[self.pairFlag]


class EqualizeHistogram(Color):

    def __init__(self, image):
        self.addParent(image)

    @cached
    def value(self):
        img = np.copy(self.parent())
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return img


class Perspective(Op):

    def __init__(self, camera, **kwargs):
        self.addParent(camera)
        self.perspectiveTransformer = laneFindingPipeline.PerspectiveTransformer(**kwargs)

    @cached
    def value(self):
        return self.perspectiveTransformer(self.parent())


class Pipeline(Op):

    def __init__(self, image):
        assert isinstance(image, Op)
        self.addParent(image)
        self.perspective = Perspective(image)
        blurred = Blur(self.perspective)
        gray = Mono(CvtColor(blurred, cv2.COLOR_RGB2GRAY))
        hls = Color(CvtColor(blurred, cv2.COLOR_RGB2HLS))
        eq = EqualizeHistogram(blurred)
        self.l_channel = ColorSplit(hls, 1)
        self.s_channel = ColorSplit(hls, 2)

        hlseq = Color(CvtColor(eq, cv2.COLOR_RGB2HLS))
        bseq_channel = Blur(ColorSplit(hlseq, 2), 71)

        self.clippedSobelS = SobelClip(self.s_channel)

        self.addParent(self.clippedSobelS)

    @cached
    def value(self):
        return self.clippedSobelS()
