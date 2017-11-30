import numpy as np
import cv2

# TODO: Remove these imports.
import laneFindingPipeline, utils

from cvflow import Op
from cvflow.misc import cached, Smol, Box


class Lambda(Op):

    def __init__(self, f, *args):
        super().__init__()
        for arg in args:
            self.addParent(arg)
        self.f = f

    @cached
    def value(self):
        args = (arg.value for arg in self.parents)
        return self.f(*args)


class Mono(Op):

    def _defaultNodeProperties(self):
        return dict(color='grey')
    
    @property
    def value(self):
        return self.parent().value


class AsMono(Mono):

    def __init__(self, parent):
        super().__init__()
        # if not hasattr(self, 'node_properties'): self.node_properties = {}
        # self.node_properties['color']  = 'red'
        self.addParent(parent)


class Boolean(Mono):
    
    def _defaultNodeProperties(self):
        return dict(color='grey', style='dashed')


class AsBoolean(Boolean):

    def __init__(self, parent):
        super().__init__()
        self.addParent(parent)


class Not(Boolean, Smol):

    def __init__(self, parent):
        super().__init__()
        self.addParent(parent)

    @property
    def value(self):
        return np.logical_not(self.parent().value)


class Color(Op):

    def _defaultNodeProperties(self):
        return dict(color='red')
    
    @cached
    def value(self):
        parent = self.parents[0]
        out = parent.value
        if isinstance(parent, Mono):
            out = np.dstack((out, out, out))
        assert len(out.shape) == 3
        return out


class AsColor(Color):

    def __init__(self, parent):
        super().__init__()
        self.addParent(parent)


class ColorSplit(Mono):

    def __init__(self, color, index):
        super().__init__()
        assert isinstance(color, Color)
        self.addParent(color)
        self.index = index

    @property
    def value(self):
        return self.parent().value[:, :, self.index]

    def __str__(self):
        return 'channel %d' % self.index


class BaseImage(Op):

    def _defaultNodeProperties(self):
        return dict(shape='box')

    def __init__(self):
        super().__init__()

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
    
    def _defaultNodeProperties(self):
        return dict(shape='box', color='red')


class MonoImage(BaseImage, Mono):
    
    node_properties = dict(shape='box', color='grey')


class Blur(Op):

    def __init__(self, parent, ksize=5):
        super().__init__()
        self.addParent(parent)
        assert ksize % 2
        self.ksize = ksize

    @cached
    def value(self):
        return cv2.GaussianBlur(self.parent().value, (self.ksize, self.ksize), 0)

    def __str__(self):
        return 'Blur with width-%d kernel.' % self.ksize


class CircleKernel(Mono):

    def __init__(self, ksize, falloff=3):
        super().__init__()
        self.ksize = ksize
        self.falloff = falloff

    @cached
    def value(self):
        kernel = cv2.getGaussianKernel(self.ksize, 0)
        kernel = (kernel * kernel.T > kernel.min() / self.falloff).astype('uint8')
        return kernel

    def __str__(self):
        return 'O kernel(%d, %s)' % (self.ksize, self.falloff)


class Dilate(Mono):

    def __init__(self, parent, kernel=5, iterations=1):
        super().__init__()
        assert isinstance(parent, Mono)
        self.addParent(AsType(parent, 'uint8'))
        if isinstance(kernel, int):
            kernel = CircleKernel(kernel)
        self.addParent(kernel)
        self.iterations = iterations

    @cached
    def value(self):
        return cv2.dilate(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations,
        )

    def __str__(self):
        return 'Dilate %d iterations.' % self.iterations


class Erode(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        super().__init__()
        assert isinstance(parent, Mono)
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        self.iterations = iterations

    @cached
    def value(self):
        return cv2.erode(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations
        )


class Opening(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        super().__init__()
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
            self.parent(0).value, self.parent(1).value, iterations=self.iterations
        )


class Sobel(Op):

    def __init__(self, channel, xy='x'):
        super().__init__()
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
        return cv2.Sobel(self.parent().value, cv2.CV_64F, *xy)

    def __str__(self):
        return 'Sobel in %s direction.' % self.xy


class _ElementwiseInequality(Boolean):

    def __init__(self, left, right, orEqualTo=False):
        super().__init__()
        self.addParent(left)
        self.addParent(right)
        self.orEqualTo = orEqualTo

    def __str__(self):
        eq = ''
        if self.orEqualTo:
            eq = '='
        return '%s %s%s %s' % (self.parents[0], self.baseSymbol, eq, self.parents[1])


class LessThan(_ElementwiseInequality):

    baseSymbol = '<'

    @cached
    def value(self):
        left, right = self.parents
        if self.orEqualTo:
            return left.value <= right.value
        else:
            return left.value < right.value


class GreaterThan(_ElementwiseInequality):

    baseSymbol = '>'

    @cached
    def value(self):
        left, right = self.parents
        if self.orEqualTo:
            return left.value >= right.value
        else:
            return left.value > right.value


class AsType(Op, Smol):

    def __init__(self, parent, kind):
        super().__init__()
        self.addParent(parent)
        self.kind = kind

    @cached
    def value(self):
        return self.parent().value.astype(self.kind)

    def __str__(self):
        return str(self.kind)


class ScalarMultiply(Op):

    def __init__(self, parent, scalar):
        super().__init__()
        self.addParent(parent)
        self.scalar = scalar

    @cached
    def value(self):
        return self.parent().value * self.scalar



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
        super().__init__()
        self.addParent(image)
        self.pairFlag = pairFlag

    @property
    def value(self):
        return cv2.cvtColor(self.parent().value, self.pairFlag)

    def __str__(self):
        return 'Convert from %s to %s.' % tuple(
            self._pairFlagsCodes[self.pairFlag].replace('COLOR_', '').split('2')
        )


class EqualizeHistogram(Color):

    def __init__(self, image):
        super().__init__()
        self.addParent(image)

    @cached
    def value(self):
        img = np.copy(self.parent().value)
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return img

    # def __str__(self)


class Perspective(Op):

    def __init__(self, camera, **kwargs):
        super().__init__()
        self.addParent(camera)
        self.perspectiveTransformer = laneFindingPipeline.PerspectiveTransformer(**kwargs)

    @cached
    def value(self):
        return self.perspectiveTransformer(self.parent().value)


class Constant(Op):

    def _defaultNodeProperties(self):
        return dict(style='dotted')

    def __init__(self, theConstant):
        super().__init__()
        self.theConstant = theConstant

    @property
    def value(self):
        return self.theConstant

    def __str__(self):
        out = str(self.theConstant).replace(':', '_')
        out = out[:50]
        return out


class And(Op, Box):

    def __init__(self, parent1, parent2):
        super().__init__()
        p1m = isinstance(parent1, Mono)
        p2m = isinstance(parent2, Mono)
        assert p1m or p2m
        self.addParent(parent1)
        self.addParent(parent2)

    @cached
    def value(self):
        p1, p2 = self.parents
        if isinstance(p1, Mono) and isinstance(p2, Mono):
            out = p1.value & p2.value
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
            out = np.copy(color.value)
            out[np.logical_not(mono.value)] = 0
        return out

    def __str__(self):
        return '%s & %s' % tuple(self.parents)


class Or(Op, Box):

    def __init__(self, parent1, parent2):
        super().__init__()
        self.addParent(parent1)
        self.addParent(parent2)

    @cached
    def value(self):
        def tomono(color):
            if len(color.shape) == 3:
                imgshape = color.shape[:2]
                mono = np.zeros(imgshape)
                for i in range(3):
                    mono = mono | color[:, :, i]
                return mono
            else:
                return color
        x1, x2 = [tomono(parent.value) for parent in self.parents]

        return x1 | x2

    def __str__(self):
        return '%s | %s' % tuple(self.parents)


class CountSeekingThresholdOp(Boolean):
    
    def __init__(self, parent, initialThreshold=150, goalCount=10000, countTol=200):
        super().__init__()
        assert isinstance(parent, Mono), '%s is not explicitly single-channel.' % parent
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
            mask = channel.value > np.ceil(threshold)
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
        