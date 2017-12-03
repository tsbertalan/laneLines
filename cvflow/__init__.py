import numpy as np
import cv2
import networkx, graphviz

from .op import *
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
        undistort = Undistort(self.input)
        perspective = Perspective(undistort)
        blurred = Blur(perspective)
        hls = CvtColor(blurred, cv2.COLOR_RGB2HLS)
        l_channel = ColorSplit(hls, 1)
        s_channel = ColorSplit(hls, 2)

        eq = EqualizeHistogram(blurred)
        hlseq = CvtColor(eq, cv2.COLOR_RGB2HLS)
        bseq_channel = Blur(ColorSplit(hlseq, 2), 71)

        lab = CvtColor(eq, cv2.COLOR_RGB2LAB)
        labaeq_channel = ColorSplit(lab, 1)
        blabbeq_channel = Blur(ColorSplit(lab, 2), 71)

        clippedSobelS = SobelClip(s_channel)

        labbeqmask = CountSeekingThreshold(blabbeq_channel)
        labbeqmask = Dilate(labbeqmask)
        
        S = clippedSobelS & labbeqmask

        clippedSobelL = SobelClip(l_channel)
        L = Dilate(clippedSobelL)

        self.output = S | L
        self.constructColorOutpout('zeros', L, S)

        self.includeInMultistep([
            clippedSobelL, clippedSobelS,
            undistort, perspective,
            hls, lab, hlseq,
            blurred, 
            eq, s_channel, l_channel,
            bseq_channel, 
            labaeq_channel, blabbeq_channel, blabbeq_channel.parent(),
            labbeqmask, labbeqmask.parent(),
            S,
            L,
        ])


class SimplePipeline(Pipeline, Boolean):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        undistort = Undistort(self.input)
        perspective = Perspective(undistort)
        blurred = Blur(perspective)
        hls = CvtColor(blurred, cv2.COLOR_RGB2HLS)
        l_channel = ColorSplit(hls, 1)
        s_channel = ColorSplit(hls, 2)
        l_binary = CountSeekingThreshold(l_channel)
        s_binary = CountSeekingThreshold(s_channel)
        markings_binary = l_binary | s_binary

        self.output = markings_binary
        self.constructColorOutpout('zeros', l_binary, s_binary)

        self.includeInMultistep([
            perspective, blurred, hls, l_channel, s_channel, 
            l_binary, s_binary, markings_binary
        ])

class MinimalPipeline(Pipeline, Boolean):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        r_channel = ColorSplit(self.input, 0)
        self.output = r_channel > 100
        self.includeInMultistep([r_channel])

class FullPipeline(Pipeline, Boolean):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        toInclude = []

        # Preprocess to get a smooth top-down view.
        undistort = Undistort(self.input)
        perspective = Perspective(undistort)
        blurred = Blur(perspective)
        preprocessings = undistort, perspective, blurred

        # Extract some colorspaces.
        hsv = CvtColor(blurred, cv2.COLOR_RGB2HSV)
        hls = CvtColor(blurred, cv2.COLOR_RGB2HLS)
        hlseq = CvtColor(EqualizeHistogram(blurred), cv2.COLOR_RGB2HLS)
        lab = CvtColor(blurred, cv2.COLOR_RGB2LAB)
        gray = CvtColor(blurred, cv2.COLOR_RGB2GRAY)
        colorspaces = hsv, hls, hlseq, lab, gray

        # Yellow-focused channels
        Y = [
            ColorSplit(hsv, 1), ColorSplit(hls, 2), ColorSplit(hlseq, 2)
        ]
        # White-focused channels
        W = [
            ColorSplit(hsv, 2), ColorSplit(hls, 1), gray, 
            -Expand(Y[2]), ColorSplit(lab, 2)
        ]

        h = lambda M: sum([hash(m) for m in M])

        # Make a bunch of permissive masks.
        permissives = {}
        for X in (Y, W):
            permissives[h(X)] = []
            for x in X:
                permissives[h(X)].append(DilateSobel(x))

        # Make the restrictive combinations.
        restrictives = {}
        for X in (Y, W):
            restrictives[h(X)] = And(*permissives[h(X)])

        # Save for later examination.
        self.permissives = permissives
        self.restrictives = restrictives

        # Also consider a conservative dynamic threshold.
        dynamic = CountSeekingThreshold(Y[0], goalCount=9000)

        # Make a color summary.
        self.constructColorOutpout(dynamic, restrictives[h(Y)], restrictives[h(W)])

        # Set the output.
        self.output = restrictives[h(Y)] | restrictives[h(W)] | dynamic

        # Collect all the Ands that haven't been claimed by other multiops,
        # and don't visualize them.
        claimed = []
        for m in self.getByKind(MultistepOp):
            claimed.extend(m.members)
        ands = self.getByKind(AndTwoInputs)
        nonincluded = [a for a in ands if a not in claimed]
        for a in ands:
            a.isVisualized = False
        self.includeInMultistep(nonincluded)

        # Set the members.
        for it in [dynamic, self.output.nparent(2)], preprocessings, colorspaces, Y, W, restrictives.values():
            toInclude.extend(it)
        for it in permissives.values():
            toInclude.extend(it)
        self.includeInMultistep(toInclude)
