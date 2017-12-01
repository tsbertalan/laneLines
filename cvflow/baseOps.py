import numpy as np
import cv2

from cvflow import Op
from cvflow.misc import cached, Circle, Box, Ellipse


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
        return dict(color='gray')
    
    @property
    def value(self):
        return self.parent().value


class AsMono(Mono):

    def __init__(self, parent):
        super().__init__()
        self.checkType(parent, Color, invert=True)
        self.addParent(parent)
        self._skipForPlot = True


class Boolean(Mono):
    
    def _defaultNodeProperties(self):
        return dict(color='gray', style='dashed')


class AsBoolean(Boolean):

    def __init__(self, parent):
        super().__init__()
        self.addParent(parent)


class Not(Boolean, Circle):

    def __init__(self, parent):
        super().__init__()
        self.addParent(parent)

    @property
    def value(self):
        return np.logical_not(self.parent().value)

    def __str__(self):
        return '!(%s)' % self.parent()


class Color(Op):

    def _defaultNodeProperties(self):
        return dict(color='blue')
    
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
        self._skipForPlot = True


class ColorSplit(Mono):

    def __init__(self, color, index):
        super().__init__()
        self.index = index
        self.checkType(color, Color)
        self.addParent(color)

    @property
    def value(self):
        return self.parent().value[:, :, self.index]

    def __str__(self):
        out = originalOut = 'channel %d' % self.index
        Target = CvtColor
        target = self.parent()
        if not isinstance(target, Target):
            target = target.parent()
        if isinstance(target, Target):
            out = str(target).split()[-1]
            if out.endswith('.'):
                out = out[:-1]
            if len(out) != 3 or len(out) <= self.index:
                out = originalOut
            else:
                out = out[self.index] + ' channel'
        return out


class ColorJoin(Color):

    def __init__(self, *channels):
        super().__init__()
        for ch in channels:
            #self.checkType(ch, Mono)
            self.addParent(ch)

    @property
    def value(self):
        return np.dstack([
            ch.value
            for ch in self.parents
        ])


class BaseImage(Op):

    def _defaultNodeProperties(self):
        return dict(shape='box')

    def __init__(self, shape=(720, 1280)):
        super().__init__()
        self.shape = shape

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
        return dict(shape='box', color='blue')


class MonoImage(BaseImage, Mono):
    
    node_properties = dict(shape='box', color='gray')


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
        return '%dx%d Gaussian blur' % (self.ksize, self.ksize)


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

    def __init__(self, mono, kernel=5, iterations=1):
        super().__init__()
        self.checkType(mono, Mono)
        mono = AsType(mono, 'uint8')

        self.addParent(mono)
        if isinstance(kernel, int):
            kernel = CircleKernel(kernel)
        else:
            if isinstance(kernel, np.ndarray):
                kernel = Constant(kernel)
        self.kernel = kernel
        self.addParent(kernel)
        self.parents[-1]._skipForPlot = True
        self.iterations = iterations

    @cached
    def value(self):
        return cv2.dilate(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations,
        )

    def __str__(self):
        return 'Dilate(iter=%d, ksize=%s)' % (self.iterations, self.kernel.value.shape)


class Erode(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        super().__init__()
        self.checkType(parent, Mono)
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
        self.checkType(parent, Mono)
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


class AsType(Op, Circle):

    def __init__(self, parent, kind, scaleUintTo255=False):
        super().__init__()
        self._skipForPlot = True
        self.addParent(parent)
        self.kind = kind
        self.scaleUintTo255 = scaleUintTo255

    @cached
    def value(self):
        inarray = self.parent().value
        if self.kind == 'uint8' or self.kind == np.uint8 and self.scaleUintTo255:
            inarray = inarray.astype('float64')
            inarray -= inarray.min()
            m = inarray.max()
            if m != 0:
                inarray /= m
                inarray *= 255
        return inarray.astype(self.kind)

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


class CvtColor(Op):

    def __init__(self, image, pairFlag):
        super().__init__()
        self.addParent(image)
        self.pairFlag = pairFlag

        pairFlags = {}
        for name in dir(cv2):
            if name.startswith('COLOR_'):
                code = getattr(cv2, name)
                if code not in pairFlags:
                    pairFlags[code] = name.upper()
                else:
                    if len(name) < len(pairFlags[code]):
                        pairFlags[code] = name.upper()
        self.flagName = pairFlags[pairFlag]

        if self.flagName.lower().endswith('gray'):
            self.node_properties.update(Mono().node_properties)
        else:
            self.node_properties.update(Color().node_properties)

    @property
    def value(self):
        return cv2.cvtColor(self.parent().value, self.pairFlag)

    def __str__(self):
        return '%s to %s.' % tuple(
            self.flagName.replace('COLOR_', '').split('2')
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


class And(Op, Ellipse):

    def __init__(self, parent1, parent2):
        super().__init__()
        Cls = Mono
        p1m = isinstance(parent1, Cls)
        p2m = isinstance(parent2, Cls)
        assert p1m or p2m, 'Either `%s` or `%s` needs to be `%s`.' % (parent1, parent2, Cls)
        self.addParent(parent1)
        self.addParent(parent2)

    @cached
    def value(self):
        p1, p2 = self.parents
        if isinstance(p1, Mono) and isinstance(p2, Mono):
            out = p1.value & p2.value
        else:
            if isinstance(p1, Mono):
                self.checkType(p2, Color)
                mono = p1
                color = p2
            else:
                self.checkType(p1, Color)
                self.checkType(p2, Mono)
                mono = p2
                color = p1
            out = np.copy(color.value)
            out[np.logical_not(mono.value)] = 0
        return out

    def __str__(self):
        return '&'

class Or(Op, Ellipse):

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
        return '|'
