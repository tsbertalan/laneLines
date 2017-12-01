import graphviz
from cvflow.baseOps import *
from cvflow.workers import *
from cvflow.misc import cached


class MultistepOp(Op):

    @property
    def input(self):
        assert hasattr(self, '_input'), 'During construction of %s, self.input is not specified.' % self
        return self._input

    @input.setter
    def input(self, op):
        # Input is not in immediate parents, 
        # since, conceptually, self is the bottom node in this subtree.
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


class DilateSobel(MultistepOp, Boolean):

    def __init__(self, singleChannel, postdilate=True, preblurksize=13, sx_thresh=20, dilate_kernel=(2, 4), dilationIterations=3):
        super().__init__()
        self.sx_thresh = sx_thresh
        self.dilate_kernel = dilate_kernel
        self.dilationIterations = dilationIterations

        self.input = singleChannel

        # Add a little *more* blurring.
        self.blur = Blur(singleChannel, ksize=preblurksize)
        self.sobelx = Sobel(self.blur, xy='x')

        # Sobel mask.
        # mask_neg = AsBoolean(AsType(LessThan(   self.sobelx, -sx_thresh), 'float32'))
        # mask_pos = AsBoolean(AsType(GreaterThan(self.sobelx,  sx_thresh), 'float32'))
        self.mask_neg = LessThan(   self.sobelx, -sx_thresh)
        self.mask_pos = GreaterThan(self.sobelx,  sx_thresh)

        kernel_midpoint = dilate_kernel[1] // 2

        # Dilate mask to the left.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, 0:kernel_midpoint] = 0
        self.dmask_neg = GreaterThan(Dilate(self.mask_neg, kernel, iterations=dilationIterations), 0.)

        # Dilate mask to the right.
        kernel = np.ones(dilate_kernel, 'uint8')
        kernel[:, kernel_midpoint:] = 0
        self.dmask_pos = GreaterThan(Dilate(self.mask_pos, kernel, iterations=dilationIterations), 0.)

        # self.sxbinary = AsBoolean(AsType(And(self.dmask_pos, self.dmask_neg), 'uint8'))
        self.sxbinary = AsBoolean(And(self.dmask_pos, self.dmask_neg))

        if postdilate:
            self.sxbinary = Dilate(self.sxbinary)

        self.output = self.sxbinary

        self.members = [
            self.sobelx, self.mask_neg, self.mask_pos, self.dmask_neg, 
            self.dmask_pos, self.sxbinary, self.blur
        ]

    @cached
    def value(self):
        return self.sxbinary.value


class SobelClip(MultistepOp, Op):

    def __init__(self, channel, threshold=None):
        super().__init__()

        self.input = channel

        # Adaptive thresholding of color.
        if threshold is None:
            threshold = CountSeekingThreshold(channel)
        self.threshold = threshold

        # Dilated masks of the threshold.
        self.narrow = Dilate(threshold, kernel=10, iterations=5)
        self.wide = Dilate(self.narrow, kernel=10, iterations=5)
        
        # Restricted Sobel-X
        self.toSobel = And(channel, Not(self.wide))

        self.sobel = DilateSobel(self.toSobel)
        self.clippedSobel = And(self.sobel, self.narrow)
        self.output = self.clippedSobel

        self.members =  [self.threshold, self.narrow, self.wide, self.toSobel, self.sobel, self.clippedSobel]

    @cached
    def value(self):
        return self.clippedSobel.value


class Pipeline(MultistepOp):

    def __init__(self, image=None, imageShape=(720, 1280)):
        super().__init__()
        if image is None:
            image = ColorImage(shape=imageShape)
        self.checkType(image, BaseImage), ''
        self.input = image
        self.input.nodeName = 'input'

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
