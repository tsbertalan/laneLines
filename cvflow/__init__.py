import numpy as np
import cv2
import networkx, graphviz

import laneFindingPipeline, utils
from . import misc

class Op:

    def __init__(self):
        if not hasattr(self, 'node_properties'):
            self.node_properties = {}
        if hasattr(self, '_defaultNodeProperties'):
            self.node_properties.update(self._defaultNodeProperties())
        self._visited = False
        self._includeSelfInMembers = True
        self._skipForPlot = False

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
            parent = baseOps.Constant(parent)
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)
        return parent

    @misc.cached
    def value(self):
        raise NotImplementedError

    def parent(self, index=0):
        return self.parents[index]

    def child(self, index=0):
        return self.children[index]

    def assembleGraph(self, d=None, currentRecursionDepth=0, format='png'):
        # if isinstance(self, CvtColor):
        #     import utils; utils.bk()
        self._visited = True

        if d is None:
            d = misc.NodeDigraph(format=format)

        if self._skipForPlot:
            assert len(self.parents) == 1
        else:
            for parent in self.parents:
                target = parent
                if target._skipForPlot:
                    assert len(target.parents) == 1
                    target = target.parent()
                d.add_edge(target, self)

        for parent in self.parents:
            if not parent._visited:
                parent.assembleGraph(d, currentRecursionDepth+1)

        for child in self.children:
            if not child._visited:
                child.assembleGraph(d, currentRecursionDepth+1)

        # Clear the visited flags so subsequent calls will work.
        if currentRecursionDepth == 0:
            self._clearVisited()

        return d

    def _clearVisited(self):
        self._visited = False
        for child in self.children:
            if child._visited:
                child._clearVisited()
        for parent in self.parents:
            if parent._visited:
                parent._clearVisited()

    def draw(self, savePath=None, format='png'):
        d = self.assembleGraph(format=format)
        if savePath is not None:
            if savePath.lower().endswith('.%s' % format.lower()):
                savePath = savePath[:-4]
            outPath = d._gv.render(savePath)
            print('Saved to %s.' % outPath)
        return d._gv

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def showValue(self, **kwargs):
        kwargs.setdefault('title', '%s $\leftarrow$ %s' % (self, tuple(self.parents)))
        utils.show(self.value, **kwargs)

from . import baseOps
from . import compositeOps

class Pipeline(compositeOps.MultistepOp):

    def __init__(self, image=None):
        super().__init__()
        if image is None:
            image = baseOps.ColorImage()
        assert isinstance(image, baseOps.BaseImage), ''
        self.input = image

    @misc.cached
    def value(self):
        return self.output.value

    def __call__(self, imgarray):
        self.input.value = imgarray
        return self.value


class ComplexPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from cvflow.baseOps import Perspective, Blur, AsColor, CvtColor, EqualizeHistogram, ColorSplit
        from cvflow.compositeOps import SobelClip

        perspective = Perspective(self.input)
        blurred = Blur(perspective)
        #gray = AsMono(CvtColor(blurred, cv2.COLOR_RGB2GRAY))
        hls = AsColor(CvtColor(blurred, cv2.COLOR_RGB2HLS))
        eq = EqualizeHistogram(blurred)
        #self.l_channel = ColorSplit(hls, 1)
        s_channel = ColorSplit(hls, 2)

        #hlseq = Color(CvtColor(eq, cv2.COLOR_RGB2HLS))
        #bseq_channel = Blur(ColorSplit(hlseq, 2), 71)

        clippedSobelS = SobelClip(s_channel)
        self.output = clippedSobelS

        self.members = [perspective, blurred, hls, eq, s_channel, clippedSobelS]


class SimplePipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from cvflow.baseOps import Perspective, Blur, AsColor, CvtColor, ColorSplit, Or, CountSeekingThresholdOp
        perspective = Perspective(self.input)
        blurred = Blur(perspective)
        hls = AsColor(CvtColor(blurred, cv2.COLOR_RGB2HLS))
        l_channel = ColorSplit(hls, 1)
        s_channel = ColorSplit(hls, 2)
        l_binary = CountSeekingThresholdOp(l_channel)
        s_binary = CountSeekingThresholdOp(s_channel)
        markings_binary = Or(l_binary, s_binary)
        self.output = markings_binary
        self.members = [perspective, blurred, hls, l_channel, s_channel, l_binary, s_binary, markings_binary, self]
