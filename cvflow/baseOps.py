import numpy as np
import cv2

from cvflow import Op
from cvflow.misc import cached

def stringFallback(naive):
    def wrapped(self):
        try:
            out = naive(self)
        except:
            out = str(type(self))
        return out
    return wrapped


class Static:

    pass


class Lambda(Op):

    def __init__(self, f, *args, **kwargs):
        for arg in args:
            self.addParent(arg)
        self.f = f
        super().__init__(**kwargs)

    @cached
    def value(self):
        args = (arg.value for arg in self.parents)
        return self.f(*args)


class Mono(Op):

    def __init__(self, **kwargs):
        self.isMono = True
        super().__init__(**kwargs)
    

class Boolean(Mono):

    def __init__(self, **kwargs):
        self.isBoolean = True
        super().__init__(**kwargs)
    

class Color(Op):

    def __init__(self, **kwargs):
        self.isColor = True
        super().__init__(**kwargs)


class ColorSplit(Mono):

    def __init__(self, color, index, **kwargs):
        self.index = index
        self.addParent(color)
        super().__init__(**kwargs)

    @property
    def value(self):
        return self.parent().value[:, :, self.index]

    @stringFallback
    def __str__(self):
        out = originalOut = 'channel %d' % self.index
        if isinstance(self.parent(), CvtColor):
            out = str(self.parent()).split()[-1]
            if len(out) != 3 or len(out) <= self.index:
                out = originalOut
            else:
                out = out[self.index] + ' channel'
        return out


class ColorJoin(Color):

    def __init__(self, *channels, **kwargs):
        for ch in channels:
            self.addParent(ch)
        super().__init__(**kwargs)

    @property
    def value(self):
        return np.dstack([
            ch.value
            for ch in self.parents
        ])


class BaseImage(Op):

    def __init__(self, shape=(720, 1280), **kwargs):
        self.shape = shape
        super().__init__(**kwargs)

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

    def __init__(self, parent, ksize=5, **kwargs):
        self.addParent(parent)
        assert ksize % 2, 'Kernel size must be odd.'
        self.ksize = ksize
        super().__init__(**kwargs)
        self.isBoolean = False

    @cached
    def value(self):
        return cv2.GaussianBlur(self.parent().value, (self.ksize, self.ksize), 0)

    @stringFallback
    def __str__(self):
        return '%dx%d Gaussian blur' % (self.ksize, self.ksize)


class UnsharpMask(Op):

    def __init__(self, parent, ksize=0, sigmax=3, **kwargs):
        self.addParent(parent)
        self.ksize = ksize
        self.sigmax = sigmax
        super().__init__(**kwargs)
        self.isBoolean = False

    @cached
    def value(self):
        image = cv2.GaussianBlur(self.parent().value, (self.ksize, self.ksize), self.sigmax)
        return cv2.addWeighted(self.parent().value, 1.5, image, -.5, 0)


class CircleKernel(Mono, Static):

    def __init__(self, ksize, falloff=3, **kwargs):
        self.ksize = ksize
        self.falloff = falloff
        super().__init__(**kwargs)

    @cached
    def value(self):
        kernel = cv2.getGaussianKernel(self.ksize, 0)
        kernel = (kernel * kernel.T > kernel.min() / self.falloff).astype('uint8')
        return kernel

    @stringFallback
    def __str__(self):
        return 'O kernel(%d, %s)' % (self.ksize, self.falloff)


def checkKernel(kernel, hidden=True):
    if isinstance(kernel, int):
        kernel = CircleKernel(kernel)
    else:
        if isinstance(kernel, np.ndarray):
            kernel = Constant(kernel)
    kernel.hidden = hidden
    return kernel


class Convolve(Mono):

    def __init__(self, parent, kernel=20, ddepth=cv2.CV_64F, **kwargs):
        self.addParent(parent)
        self.kernel = self.addParent(checkKernel(kernel))
        self.ddepth = ddepth
        super().__init__(**kwargs)

    @cached
    def value(self):
        return cv2.filter2D(
            self.parent().value, self.ddepth, self.kernel.value
        )

    @stringFallback
    def __str__(self):
        return '%s convolution' % self.kernel


class Dilate(Mono):

    def __init__(self, mono, kernel=5, iterations=1, **kwargs):
        mono = AsType(mono, 'uint8')
        self.addParent(mono)
        self.kernel = self.addParent(checkKernel(kernel))
        self.parents[-1].hidden = True
        self.iterations = iterations
        super().__init__(**kwargs)

    @cached
    def value(self):
        return cv2.dilate(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations,
        )

    @stringFallback
    def __str__(self):
        return 'Dilate(iter=%d, ksize=%s)' % (self.iterations, self.kernel.value.shape)


class Erode(Mono):

    def __init__(self, parent, kernel=None, iterations=1, **kwargs):
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        self.iterations = iterations
        super().__init__(**kwargs)

    @cached
    def value(self):
        return cv2.erode(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations
        )


class Opening(Mono):

    def __init__(self, parent, kernel=None, iterations=1, **kwargs):
        self.iterations = iterations
        self.addParent(parent)
        if kernel is None:
            kerenel = CircleKernel(5)
        self.addParent(kernel)
        super().__init__(**kwargs)

    @cached
    def value(self):
        return cv2.morphologyEx(
            self.parent(0).value, self.parent(1).value, iterations=self.iterations
        )


class Sobel(Op):

    def __init__(self, channel, xy='x', ddepth=cv2.CV_64F, **kwargs):
        self.addParent(channel)
        self.xy = xy
        self.ddepth = ddepth
        super().__init__(**kwargs)

    @cached
    def value(self):
        if self.xy == 'x':
            xy = (1, 0)
        elif self.xy == 'xy':
            xy = (1, 1)
        else:
            xy = (0, 1)
        return cv2.Sobel(self.parent().value, self.ddepth, *xy)

    @stringFallback
    def __str__(self):
        return 'Sobel in %s direction.' % self.xy


class _ElementwiseInequality(Boolean):

    def __init__(self, left, right, orEqualTo=False, **kwargs):
        self.addParent(left)
        self.addParent(right)
        self.orEqualTo = orEqualTo
        super().__init__(**kwargs)

    @stringFallback
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


class EqualTo(Boolean):

    baseSymbol = '=='

    def __init__(self, left, right, orEqualTo=False, **kwargs):
        self.addParent(left)
        self.addParent(right)
        super().__init__(**kwargs)

    @stringFallback
    def __str__(self):
        return '%s %s %s' % (self.parents[0], self.baseSymbol, self.parents[1])

    @cached
    def value(self):
        left, right = self.parents
        return left.value == right.value


class AsType(Op):

    def __init__(self, parent, kind, scaleUintTo255=False, **kwargs):
        self.hidden = True
        self.addParent(parent)
        self.kind = kind
        self.scaleUintTo255 = scaleUintTo255
        super().__init__(**kwargs)
        self.node_properties['shape'] = 'circle'

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
        elif self.scaleUintTo255:
            raise ValueError('scaleUintTo255=True was passed, but dtype is not recognized as  uint8.')
        return inarray.astype(self.kind)

    @stringFallback
    def __str__(self):
        return str(self.kind)


class ScalarMultiply(Op):

    def __init__(self, parent, scalar, **kwargs):
        self.addParent(parent)
        self.scalar = scalar
        super().__init__(**kwargs)

    @cached
    def value(self):
        return self.parent().value * self.scalar


class CvtColor(Op):

    def __init__(self, image, pairFlag, **kwargs):
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

        super().__init__(**kwargs)

        if self.flagName.lower().endswith('gray'):
            self.isMono = True
            self.isColor = False
            self.node_properties.update(Mono().node_properties)
        else:
            self.isMono = False
            self.isColor = True
            self.node_properties.update(Color().node_properties)

    @property
    def value(self):
        return cv2.cvtColor(self.parent().value, self.pairFlag)

    @stringFallback
    def __str__(self):
        return '%s to %s' % tuple(
            self.flagName.replace('COLOR_', '').split('2')
        )


class EqualizeHistogram(Color):

    def __init__(self, image, **kwargs):
        self.addParent(image)
        super().__init__(**kwargs)

    @cached
    def value(self):
        img = np.copy(self.parent().value)
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i])
        return img


class Constant(Op, Static):

    def __init__(self, theConstant, **kwargs):
        self.theConstant = theConstant
        if isinstance(theConstant, (int, float)):
            self.isScalar = True
        self.node_properties['style'] = 'dotted'
        self.hidden = True
        super().__init__(**kwargs)

    @property
    def shape(self):
        return self.theConstant.shape

    @property
    def value(self):
        return self.theConstant

    @stringFallback
    def __str__(self):
        if isinstance(self.theConstant, np.ndarray):
            what = 'ndarray'
        else:
            what = self.theConstant
        out = str(what).replace(':', '_')
        out = out[:50]
        return out


class Arithmetic(Op):

    def __init__(self, **kwargs):
        self.isArithmetic = True
        super().__init__(**kwargs)  


class Divide(Arithmetic):

    def __init__(self, left, right, zeroSpotsToMax=True, **kwargs):
        self.addParent(left)
        self.addParent(right)
        self.zeroSpotsToMax = zeroSpotsToMax
        super().__init__(**kwargs)  

    @cached
    def value(self):
        left = self.parent(0).value
        right = np.copy(self.parent(1).value)
        right[right==0] = min(right[right!=0])
        return left / right

        # res = left / right
        # maxOk = res[right != 0].max()
        # res[right == 0] = maxOk
        # res[(right == 0) & (left == 0)] = 0
        # return res

    @stringFallback
    def __str__(self):
        return '//' if self.zeroSpotsToMax else '/'


class Abs(Arithmetic):

    def __init__(self, parent, **kwargs):
        self.addParent(parent)
        super().__init__(**kwargs)

    @cached
    def value(self):
        return abs(self.parent().value)


class Add(Arithmetic):

    def __init__(self, *parents, **kwargs):
        for parent in parents:
            self.addParent(parent)
        super().__init__(**kwargs)

    @cached
    def value(self):
        out = 0
        for parent in self.parents:
            out += parent.value
        return out


class Subtract(Arithmetic):

    def __init__(self, left, right, **kwargs):
        self.addParent(left)
        self.addParent(right)
        super().__init__(**kwargs)

    @cached
    def value(self):
        return self.parent(0).value - self.parent(1).value


class Multiply(Arithmetic):

    def __init__(self, *parents, **kwargs):
        for parent in parents:
            self.addParent(parent)
        super().__init__(**kwargs)

    @cached
    def value(self):
        out = 1
        for parent in self.parents:
            try:
                out = parent.value * out
            except TypeError:
                out *= parent.value
        return out


class Pow(Arithmetic):

    def __init__(self, x, power, **kwargs):
        self.addParent(x)
        self.addParent(power)
        super().__init__(**kwargs)


    @cached
    def value(self):
        return self.parent(0).value ** self.parent(1).value


class Scalar(Arithmetic):

    def __init__(self, **kwargs):
        self.isScalar = True
        super().__init__(**kwargs)


class Max(Scalar):

    def __init__(self, op, **kwargs):
        self.addParent(op)
        super().__init__(**kwargs)


    @cached
    def value(self):
        x = self.parent().value
        try:
            return x.max()
        except AttributeError:
            return max(x)


class Min(Scalar):

    def __init__(self, op, **kwargs):
        self.addParent(op)
        super().__init__(**kwargs)


    @cached
    def value(self):
        x = self.parent().value
        try:
            return x.min()
        except AttributeError:
            return min(x)


class Logical(Op):

    def __init__(self, **kwargs):
        self.isLogical = True
        super().__init__(**kwargs)  


class Not(Logical):

    def __init__(self, parent, **kwargs):
        self.addParent(parent)
        super().__init__(**kwargs)
        self.isBoolean = True

    @property
    def value(self):
        return np.logical_not(self.parent().value)

    @stringFallback
    def __str__(self):
        return '!(%s)' % self.parent()


class And(Logical):

    def __init__(self, p1, p2, **kwargs):
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
                got = p.isBoolean
                if got:
                    msg += ' %s.%s = %s' % (n(p), attr, got)
            raise AssertionError(msg + '.')
        super().__init__(**kwargs)

    @cached
    def value(self):
        p1, p2 = self.parents
        if p1.isBoolean and p2.isBoolean:
            out = p1.value & p2.value
        else:
            if p1.isBoolean:
                mask = p1
                masked = p2 
            elif p2.isBoolean:
                mask = p2
                masked = p1
            else:
                raise ValueError(
                    '%s, but need one or both argument to be Boolean.' %
                    actualTypes()
                )
            out = np.copy(masked.value)
            out[np.logical_not(mask.value)] = 0
        return out

    @stringFallback
    def __str__(self):
        return '&'


class Or(Logical):

    def __init__(self, parent1, parent2, **kwargs):
        self.addParent(parent1)
        self.addParent(parent2)
        super().__init__(**kwargs)

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

    @stringFallback
    def __str__(self):
        return '|'
