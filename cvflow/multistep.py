import graphviz

from cvflow import Op
from cvflow.misc import cached
from cvflow.baseOps import *

class PassThrough(Op, Circle):

    def __init__(self, input):
        super().__init__()
        if hasattr(self, '_hidden'):
            del self._hidden
        self.addParent(input)
        self.node_properties['color'] = 'blue'

    @property
    def value(self):
        return self.parent().value

    @value.setter
    def value(self, newimage):
        self.parent().value = newimage

    @property
    def shape(self):
        return self.parent().shape

    def __str__(self):
        return ''

    @property
    def hidden(self):
        if hasattr(self, '_hidden'):
            return self._hidden
        else:
            return len(self.parents) == 1 and len(self.children) == 1

    @hidden.setter
    def hidden(self, fixedValue):
        self._hidden = fixedValue


class Input(PassThrough): pass


class Output(Input): pass


class MultistepOp(Op):

    def __init__(self):
        super().__init__()
        self.node_properties['shape'] = 'parallelogram'

    @property
    def input(self):
        assert hasattr(self, '_input'), 'During construction of %s, self.input is not specified.' % self
        return self._input

    @input.setter
    def input(self, op):
        # Add a No-op so the input only comes in on one line,
        # even if it's used more than once.
        input = Input(op)
        #self.addParent(input)
        self._input = input

    @property
    def output(self):
        assert hasattr(self, '_output'), 'During construction of %s, self.output is not specified.' % self
        return self._output

    @output.setter
    def output(self, op):
        output = Output(op)
        self.addParent(output)
        self._output = output

    @cached
    def value(self):
        return self.output.value

    def assembleGraph(self, *args, **kwargs):
        # Do the normal graph construction recursion.
        d = super().assembleGraph(*args, **kwargs)

        # Add subgraph with members of this op.
        members = self.members
        graph_attr = dict(shape='box', label=str(self))
        for k in 'color', 'style':
            if k in self.node_properties:
                graph_attr[k] = self.node_properties[k]
        label = d.add_subgraph(members, str(self), graph_attr=graph_attr)
        if not hasattr(self, 'nodeName'):
            self.nodeName = label
            d.add_node(self)

        return d

    def includeInMultistep(self, members, hiddenClasses=[Constant,]):
        out = []
        for m in members:
            out.append(m)
            if not isinstance(m, MultistepOp):
                out.extend(m.parents)
        for m in out:
            for Hidden in hiddenClasses:
                if isinstance(m, Hidden):
                    m.hidden = True
        self.members = out

    @property
    def members(self):
        if not hasattr(self, '_members'): self._members = []
        members = self._members

        if self._includeSelfInMembers:
            if self not in members:
                members.append(self)

        # Remove any nodes explicitly disincluded.
        members.append(self.input)
        members.append(self.output)
        members = [m for m in members if not m.hidden]

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
        image.nodeName = 'Input'
        self.nodeName = 'Output'

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
                c.hidden = True

        self.colorOutput = ColorJoin(*channels)
        self.colorOutput.nodeName = 'color output'
        self.members = [self.colorOutput]
