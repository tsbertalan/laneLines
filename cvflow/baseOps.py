import numpy as np
import cv2

from cvflow import Op
from cvflow.misc import cached


class Lambda(Op):

    def __init__(self, f, *args):
        for arg in args:
            self.addParent(arg)
        self.f = f
        super().__init__()

    @cached
    def value(self):
        args = (arg.value for arg in self.parents)
        return self.f(*args)


class Mono(Op):

    def __init__(self):
        self.isMono = True
        super().__init__()
    

class Boolean(Mono):

    def __init__(self):
        self.isBoolean = True
        super().__init__()
    

class Color(Op):

    def __init__(self):
        self.isColor = True
        super().__init__()


class Logical(Op):

    def __init__(self):
        self.isLogical = True
        super().__init__()  


class ColorSplit(Mono):

    def __init__(self, color, index):
        self.index = index
        self.addParent(color)
        super().__init__()

    @property
    def value(self):
        return self.parent().value[:, :, self.index]

    def __str__(self):
        out = originalOut = 'channel %d' % self.index
        if isinstance(self.parent(), CvtColor):
            out = str(self.parent()).split()[-1]
            if out.endswith('.'):
                out = out[:-1]
            if len(out) != 3 or len(out) <= self.index:
                out = originalOut
            else:
                out = out[self.index] + ' channel'
        return out


class ColorJoin(Color):

    def __init__(self, *channels):
        for ch in channels:
            self.addParent(ch)
        super().__init__()

    @property
    def value(self):
        return np.dstack([
            ch.value
            for ch in self.parents
        ])


class BaseImage(Op):

    def __init__(self, shape=(720, 1280)):
        self.shape = shape
        super().__init__()

    @property
    def value(self):
        try:
            return self.image
        except AttributeError:
            raise RuntimeError('self.value for `%s` is not yet initialized.' % self)

    @value.setter
    def value(self, newimage):
        self.image = newimage
        if self.isMono:
            assert len(newimage.shape) == 2
        else:
            assert len(newimage.shape) == 3
        self.invalidateCache()


class ColorImage(BaseImage):
    
    def __init__(self, **kwargs):
        self.isColor = True
        super().__init__(**kwargs)


class MonoImage(BaseImage):
    
    def __init__(self, **kwargs):
        self.isMono = True
        super().__init__(**kwargs)


class Blur(Op):

    def __init__(self, parent, ksize=5):
        self.addParent(parent)
        assert ksize % 2
        self.ksize = ksize
        super().__init__()

    @cached
    def value(self):
        return cv2.GaussianBlur(self.parent().value, (self.ksize, self.ksize), 0)

    def __str__(self):
        return '%dx%d Gaussian blur' % (self.ksize, self.ksize)


class CircleKernel(Mono):

    def __init__(self, ksize, falloff=3):
        self.ksize = ksize
        self.falloff = falloff
        super().__init__()

    @cached
    def value(self):
        kernel = cv2.getGaussianKernel(self.ksize, 0)
        kernel = (kernel * kernel.T > kernel.min() / self.falloff).astype('uint8')
        return kernel

    def __str__(self):
        return 'O kernel(%d, %s)' % (self.ksize, self.falloff)


class Dilate(Mono):

    def __init__(self, mono, kernel=5, iterations=1):
        mono = AsType(mono, 'uint8')
        self.addParent(mono)
        if isinstance(kernel, int):
            kernel = CircleKernel(kernel)
        else:
            if isinstance(kernel, np.ndarray):
                kernel = Constant(kernel)
        self.addParent(kernel)

        self.kernel = kernel
        self.parents[-1].hidden = True
        self.iterations = iterations
        super().__init__()

    @cached
    def value(self):
        return cv2.dilate(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations,
        )

    def __str__(self):
        return 'Dilate(iter=%d, ksize=%s)' % (self.iterations, self.kernel.value.shape)


class Erode(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        self.iterations = iterations
        super().__init__()

    @cached
    def value(self):
        return cv2.erode(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations
        )


class Opening(Mono):

    def __init__(self, parent, kernel=None, iterations=1):
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        super().__init__()
        self.iterations = iterations

    @cached
    def value(self):
        return cv2.morphologyEx(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations
        )


class Sobel(Op):

    def __init__(self, channel, xy='x'):
        self.addParent(channel)
        self.xy = xy
        super().__init__()

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
        self.addParent(left)
        self.addParent(right)
        self.orEqualTo = orEqualTo
        super().__init__()

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


class AsType(Op):

    def __init__(self, parent, kind, scaleUintTo255=False):
        self.hidden = True
        self.addParent(parent)
        self.kind = kind
        self.scaleUintTo255 = scaleUintTo255
        self.node_properties['shape'] = 'circle'
        super().__init__()

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
        self.addParent(parent)
        self.scalar = scalar
        super().__init__()

    @cached
    def value(self):
        return self.parent().value * self.scalar


class CvtColor(Op):

    def __init__(self, image, pairFlag):
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
            self.isMono = True
            self.isColor = False
            self.node_properties.update(Mono().node_properties)
        else:
            self.isMono = False
            self.isColor = True
            self.node_properties.update(Color().node_properties)

        super().__init__()

    @property
    def value(self):
        return cv2.cvtColor(self.parent().value, self.pairFlag)

    def __str__(self):
        return '%s to %s.' % tuple(
            self.flagName.replace('COLOR_', '').split('2')
        )


class EqualizeHistogram(Color):

    def __init__(self, image):
        self.addParent(image)
        super().__init__()

    @cached
    def value(self):
        img = np.copy(self.parent().value)
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return img


class Constant(Op):

    def __init__(self, theConstant):
        self.theConstant = theConstant
        self.node_properties['style'] = 'dotted'
        super().__init__()

    @property
    def shape(self):
        return self.theConstant.shape

    @property
    def value(self):
        return self.theConstant

    def __str__(self):
        if isinstance(self.theConstant, np.ndarray):
            what = 'ndarray'
        else:
            what = self.theConstant
        out = str(what).replace(':', '_')
        out = out[:50]
        return out


class Not(Logical):

    def __init__(self, parent):
        self.addParent(parent)
        super().__init__()
        self.isBoolean = True

    @property
    def value(self):
        return np.logical_not(self.parent().value)

    def __str__(self):
        return '!(%s)' % self.parent()


class And(Logical):

    def __init__(self, p1, p2):
        self.addParent(p1)
        self.addParent(p2)
        if not (
            (p1.isMono and p2.isMono)
            or (p1.isColor and p2.isMono)
            or (p2.isColor and p1.isMono)
            ):

            n = lambda p: type(p).__name__
            msg = 'Attempted (%s & %s) with' % tuple([n(p) for p in (p1, p2)])
            for p in (p1, p2):
                for attr in 'isMono', 'isColor':
                    got = getattr(p, attr)
                    if got:
                        msg += ' %s.%s = %s' % (n(p), attr, got)
            raise AssertionError(msg + '.')
        super().__init__()

    @cached
    def value(self):
        p1, p2 = self.parents
        if p1.isMono and p2.isMono:
            out = p1.value & p2.value
        else:
            def chk(mono, color):
                if not color.isColor:
                    raise ValueError('%s, but `%s` is Mono, so `%s` must be Color or also Mono.' % (
                        actualTypes(), mono.getSimpleName(), color.getSimpleName()
                    ))
            if p1.isMono:
                chk(p1, p2)
                mono = p1
                color = p2
            elif p2.isMono:
                chk(p2, p1)
                mono = p2
                color = p1
            else:
                raise ValueError('%s, but neither type was Mono.' % actualTypes())
            out = np.copy(color.value)
            out[np.logical_not(mono.value)] = 0
        return out

    def __str__(self):
        return '&'


class Or(Logical):

    def __init__(self, parent1, parent2):
        self.addParent(parent1)
        self.addParent(parent2)
        super().__init__()

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
