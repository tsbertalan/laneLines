import graphviz
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
        self._input = op

    @property
    def output(self):
        assert hasattr(self, '_output'), 'During construction of %s, self.output is not specified.' % self
        return self._output

    @output.setter
    def output(self, op):
        self.addParent(op)
        self._output = op

    def assembleGraph(self, *args, **kwargs):
        d = super().assembleGraph(*args, **kwargs)

        if hasattr(self, 'members'):
            graph_attr = dict(shape='box', label=str(self))
            for k in 'color', 'style':
                if k in self.node_properties:
                    graph_attr[k] = self.node_properties[k]
            sg = graphviz.Digraph(name='cluster %s' % self, graph_attr=graph_attr)
            members = self.members

            if self._includeSelfInMembers:
                if self not in members:
                    members.append(self)

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

class DilateSobel(MultistepOp, Boolean):

    def __init__(self, singleChannel, postdilate=True, preblurksize=13, sx_thresh=20, dilate_kernel=(2, 4), dilationIterations=3):
        super().__init__()
        self.sx_thresh = sx_thresh
        self.dilate_kernel = dilate_kernel
        self.dilationIterations = dilationIterations

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
            self.dmask_pos, self.sxbinary, self.blur, self
        ]
        for m in self.members:
            if (
                m not in [self.blur]
                and len(m.parents) > 0
            ):
                self.members.append(m.parents[0])

    @cached
    def value(self):
        return self.sxbinary.value


class SobelClip(MultistepOp, Op):

    def __init__(self, channel, threshold=None):
        super().__init__()

        # Adaptive thresholding of color.
        if threshold is None:
            threshold = CountSeekingThresholdOp(channel)
        self.threshold = threshold

        # Dilated masks of the threshold.
        self.narrow = Dilate(threshold, kernel=10, iterations=5)
        self.wide = Dilate(self.narrow, kernel=10, iterations=5)
        
        # Restricted Sobel-X
        self.toSobel = And(channel, Not(self.wide))

        self.sobel = DilateSobel(self.toSobel)
        self.clippedSobel = And(self.sobel, self.narrow)
        self.output = self.clippedSobel

        self.members =  [self.threshold, self.narrow, self.wide, self.toSobel, self.sobel, self.clippedSobel, self]

    @cached
    def value(self):
        return self.clippedSobel.value

