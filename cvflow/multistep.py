import graphviz

from cvflow import Op
from cvflow.misc import cached
from cvflow.baseOps import *

class MultistepOp(Op):

    @property
    def input(self):
        assert hasattr(self, '_input'), 'During construction of %s, self.input is not specified.' % self
        return self._input

    @input.setter
    def input(self, op):
        # Input is not in immediate parents, 
        # since, conceptually, self is the bottom node in this subtree.
        op.nodeName = 'input'
        self._input = op

    @property
    def output(self):
        assert hasattr(self, '_output'), 'During construction of %s, self.output is not specified.' % self
        return self._output

    @output.setter
    def output(self, op):
        op.nodeName = 'output'
        self.addParent(op)
        self._output = op

    def assembleGraph(self, *args, **kwargs):
        # Do the normal graph construction recursion.
        d = super().assembleGraph(*args, **kwargs)

        # Add subgraph with members of this op.
        members = self.members
        graph_attr = dict(shape='box', label=str(self))
        for k in 'color', 'style':
            if k in self.node_properties:
                graph_attr[k] = self.node_properties[k]
        sg = graphviz.Digraph(name='cluster %s' % self, graph_attr=graph_attr)
        for member in members:
            sg.node(d._nid(member))
        d._gv.subgraph(sg)

        return d

    @property
    def members(self):
        if not hasattr(self, '_members'): self._members = []
        members = self._members

        if self._includeSelfInMembers:
            if self not in members:
                members.append(self)

        # Add the 'uint8's for Dilate etc.
        for member in members:
            
            # Look at the immediate parents of our first-tier members.
            for parent in member.parents:
                if parent is not self.input and parent not in members:
                    # The first grandparent is a member.
                    grandparents = parent.parents
                    if len(grandparents) > 0 and grandparents[0] in members:
                            members.append(parent)
                    elif not isinstance(parent, BaseImage):
                        members.append(parent)

        # Remove any nodes explicitly disincluded.
        members = [m for m in members if not m._skipForPlot and m is not self.input]

        return members

    @members.setter
    def members(self, newmembers):
        if not hasattr(self, '_members'): self._members = []
        self._members.extend(newmembers)


class Pipeline(MultistepOp):

    def __init__(self, image=None, imageShape=(720, 1280)):
        super().__init__()
        if image is None:
            image = ColorImage(shape=imageShape)
        self.checkType(image, BaseImage), ''
        self.input = image

    @cached
    def value(self):
        return self.output.value

    def __call__(self, imgarray, color=False):
        self.input.value = imgarray
        if color:
            if hasattr(self, 'colorOutput'):
                return self.colorOutput.value
            else:
                from warnings import warn
                warn('`%s` called with color=True, but does not implement colorOutput.' % self)
        return self.value

    def constructColorOutpout(self, r, b, g, dtype='uint8', scaleUintTo255=True):
        channels = [
            Constant(np.zeros(self.input.shape).astype(dtype))
            if c == 'zeros'
            else AsType(c, dtype, scaleUintTo255=scaleUintTo255)
            for c in (r,b,g)
        ]
        for c in channels:
            if isinstance(c, Constant):
                c._skipForPlot = True

        self.colorOutput = ColorJoin(*channels)
        self.colorOutput.nodeName = 'color output'
        self.members = [self.colorOutput]
