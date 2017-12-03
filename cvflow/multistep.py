import graphviz

import cvflow
from cvflow import Op
from cvflow.misc import cached
from cvflow.baseOps import *

class PassThrough(Op):

    def __init__(self, input, **kwargs):
        self.addParent(input)
        if hasattr(self, '_hidden'):
            del self._hidden
        self.node_properties['shape'] = 'none'
        self.copySetProperties(input)
        super().__init__(**kwargs)
        self.isPassThrough = True

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

    def __init__(self, **kwargs):
        self.isMultistepOp = True
        super().__init__(**kwargs)

    @property
    def input(self):
        assert hasattr(self, '_input'), 'During construction of %s, self.input is not specified.' % self
        return self._input

    @input.setter
    def input(self, op):
        # Add a No-op so the input only comes in on one line,
        # even if it's used more than once.
        name = op.getSimpleName()
        op.nodeName = '%s (to %s)' % (name, self.getSimpleName())
        op.node_properties['color'] = 'blue'
        input = Input(op)
        input.nodeName = 'Input to %s (%s).' % (self.getSimpleName(), name)
        if not isinstance(self, Pipeline):
            self.addParent(op)
        self._input = input

    @property
    def output(self):
        assert hasattr(self, '_output'), 'During construction of %s, self.output is not specified.' % self
        return self._output

    @output.setter
    def output(self, op):
        name = op.getSimpleName()
        op.nodeName = lambda : 'Output from %s: %s' % (self.getSimpleName(), name)
        output = Output(op)
        output.nodeName = lambda : 'Output %s from %s.' % (name, self.getSimpleName())
        output.hidden = True
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
        if self.nodeName is None:
            self.nodeName = label
            d.add_node(self)

        return d

    def includeInMultistep(self, members, hiddenClasses=[Constant,]):
        # TODO: Automate this.
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

        # Specifically include the guards.
        members.append(self.input)
        members.append(self.output)

        # And of course the output's input.
        members.append(self.output.parent())

        # Remove any nodes explicitly disincluded.
        members = [m for m in members if not m.hidden]

        # Remove duplicates and return an unsorted list.
        return set(members)

    @members.setter
    def members(self, newmembers):
        if not hasattr(self, '_members'): self._members = []
        self._members.extend(newmembers)

    def getMembersByType(self, Kind):
        return [m for m in self.members if isinstance(m, Kind)]

    @property
    def toposortMembers(self):
        toposort = self.assembleGraph().toposort()
        return [m for m in toposort if m in self.members]

    def showMembers(self, 
        which='all', axes=None, titleSize=10, subplotKwargs={},
        excludeTypes=[Input],
        showMultistepParents=True, titleColor='black',
        _getColor=None,
        **kwargs):
        for k, v in dict(top=.9, bottom=0, left=0, right=1, wspace=.01, hspace=.17).items():
            kwargs.setdefault(k, v)
        if which == 'all':
            which = self.members
        # Order members topologically.
        which = [
            m for m in self.toposortMembers
            if m in which and type(m) not in excludeTypes
            and not m.isScalar
        ]
        if len(which) == 0:
            return []

        if axes is None:
            axes = cvflow.misc.axesGrid(len(which), clearAxisTicks=True, **subplotKwargs).ravel()
        assert len(which) <= len(axes)

        multistepColorSources = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan']
        if _getColor is None:
            def _getColor():
                out = multistepColorSources[_getColor.i]
                _getColor.i += 1
                if _getColor.i == len(multistepColorSources):
                    _getColor.i = 0
                return out
            _getColor.i = 0

        multistepColors = {}

        for op, ax in zip(which, axes):
            op.showValue(ax=ax, **kwargs)

            if isinstance(op, MultistepOp):
                color = _getColor()
                multistepColors[op] = color
                for side in 'left', 'right', 'top', 'bottom':
                    ax.spines[side].set_color(color)
                    ax.spines[side].set_linewidth(4)
                ax.set_frame_on(True)

            if len(op.children) == 1 and isinstance(op.child(), cvflow.Output):
                ax.title.set_color(titleColor)

            if titleSize is not None:
                ax.title.set_fontsize(titleSize)

        fig = axes[0].figure
        if titleColor != 'black':
            #fig.frameon(True)
            fig.patch.set_edgecolor(titleColor)
            fig.patch.set_linewidth(4)
        fig.suptitle(str(self) if self.nodeName is None else self.nodeName, color=titleColor)
        shown = [fig]

        if showMultistepParents:
            for p in [p for p in which if isinstance(p, cvflow.MultistepOp)]:
                shown.extend(p.showMembers(
                    which='all', subplotKwargs=subplotKwargs, 
                    excludeTypes=excludeTypes, showMultistepParents=True, 
                    titleColor=multistepColors[p], _getColor=_getColor,
                    **kwargs
                ))

        return shown


class Pipeline(MultistepOp):

    def __init__(self, image=None, imageShape=(720, 1280), **kwargs):
        """

        In subclasses where this constructor is called as `super(**kwargs).__init__()`, 
        that call should come *after* the subclass is done initializting
        (so `self.input` gets set). Subclasses should not call `self.addParent(op)`.
        """
        if image is None:
            image = ColorImage(shape=imageShape)
        image.hidden = True
        self.input = image
        # self.nodeName = 'Pipeline output'
        super().__init__(**kwargs)

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

    @cvflow.constantOrOpInput
    def constructColorOutpout(self, *args, **kwargs):
        self.colorOutput = super().constructColorOutpout(*args, **kwargs)
        self.colorOutput.nodeName = 'Color pipeline output'
        self.members = [self.colorOutput]
        return self.colorOutput
