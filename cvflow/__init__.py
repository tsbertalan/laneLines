import numpy as np
import cv2
import networkx, graphviz

import laneFindingPipeline, utils
from cvflow.misc import *

class Op:

    def __init__(self):
        if not hasattr(self, 'node_properties'):
            self.node_properties = {}
        if hasattr(self, '_defaultNodeProperties'):
            self.node_properties.update(self._defaultNodeProperties())
        self._visited = False

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
        # if isinstance(self, CvtColor):
        #     import utils; utils.bk()
        self._visited = True
        if d is None:
            d = NodeDigraph()
        d.add_node(self)
        for child in self.children:
            if not d.containsEdge(self, child):
                d.add_edge(self, child)
            if not child._visited:
                child.assembleGraph(d, currentRecursionDepth+1)
        for parent in self.parents:
            if not d.containsEdge(parent, self):
                d.add_edge(parent, self)
            if not parent._visited:
                parent.assembleGraph(d, currentRecursionDepth+1)

        if hasattr(self, 'members'):
            sg = graphviz.Digraph(name='cluster %s' % self, graph_attr=dict(shape='box', label=str(self)))
            members = self.members

            # Add the 'uint8's for Dilate etc.
            for member in members:
                for parent in member.parents:
                    if parent not in members:
                        if len(parent.parents) > 0:
                            if parent.parents[0] in members:
                                members.append(parent)
                        elif not isinstance(parent, BaseImage):
                            members.append(parent)

            for member in members:
                sg.node(d._nid(member))
            d._gv.subgraph(sg)
        return d

    def draw(self):
        d = self.assembleGraph()
        return d._gv

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def showValue(self, **kwargs):
        kwargs.setdefault('title', '%s $\leftarrow$ %s' % (self, tuple(self.parents)))
        utils.show(self.value, **kwargs)


class Pipeline(Op):

    def __init__(self, image):
        super().__init__()
        assert isinstance(image, Op)
        self.addParent(image)
        self.perspective = Perspective(image)
        blurred = Blur(self.perspective)
        #gray = AsMono(CvtColor(blurred, cv2.COLOR_RGB2GRAY))
        hls = AsColor(CvtColor(blurred, cv2.COLOR_RGB2HLS))
        eq = EqualizeHistogram(blurred)
        #self.l_channel = ColorSplit(hls, 1)
        self.s_channel = ColorSplit(hls, 2)

        #hlseq = Color(CvtColor(eq, cv2.COLOR_RGB2HLS))
        #bseq_channel = Blur(ColorSplit(hlseq, 2), 71)

        self.clippedSobelS = SobelClip(self.s_channel)

        self.addParent(self.clippedSobelS)

    @cached
    def value(self):
        return self.clippedSobelS()


from cvflow.baseOps import *
from cvflow.compositeOps import *
