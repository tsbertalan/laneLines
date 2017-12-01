import numpy as np
import cv2
import networkx, graphviz

from .op import Op
from . import misc
from . import baseOps
from . import workers
from . import multistep
from . import compositeOps
from .baseOps import *
from .workers import *
from .multistep import *
from .compositeOps import *


class ComplexPipeline(Pipeline, Boolean):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        perspective = Perspective(self.input)
        blurred = Blur(perspective)
        hls = AsColor(CvtColor(blurred, cv2.COLOR_RGB2HLS))
        eq = EqualizeHistogram(blurred)
        l_channel = ColorSplit(hls, 1)
        s_channel = ColorSplit(hls, 2)

        hlseq = AsColor(CvtColor(eq, cv2.COLOR_RGB2HLS))
        bseq_channel = Blur(ColorSplit(hlseq, 2), 71)

        lab = AsColor(CvtColor(eq, cv2.COLOR_RGB2LAB))
        labaeq_channel = ColorSplit(lab, 1)
        blabbeq_channel = AsMono(Blur(ColorSplit(lab, 2), 71))

        clippedSobelS = SobelClip(s_channel)

        labbeqmask = CountSeekingThreshold(blabbeq_channel)
        labbeqmask = Dilate(labbeqmask)
        
        S = clippedSobelS & labbeqmask

        clippedSobelL = SobelClip(l_channel)
        L = Dilate(clippedSobelL)

        self.output = S | L

        self.constructColorOutpout('zeros', L, S)
        self.members = [
            perspective, blurred, 
            hls, eq, s_channel, l_channel,
            hlseq, bseq_channel, 
            lab, labaeq_channel, blabbeq_channel,
            labbeqmask,
            clippedSobelS, S,
            clippedSobelL, L,
            self.output,
        ]


class SimplePipeline(Pipeline, Boolean):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        undistort = Undistort(self.input)
        perspective = Perspective(undistort)
        blurred = Blur(perspective)
        hls = AsColor(CvtColor(blurred, cv2.COLOR_RGB2HLS))
        l_channel = ColorSplit(hls, 1)
        s_channel = ColorSplit(hls, 2)
        l_binary = CountSeekingThreshold(l_channel)
        s_binary = CountSeekingThreshold(s_channel)
        markings_binary = l_binary | s_binary
        self.output = markings_binary
        self.constructColorOutpout('zeros', l_binary, s_binary)
        self.members = [perspective, blurred, hls, l_channel, s_channel, l_binary, s_binary, markings_binary]
