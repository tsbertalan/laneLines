import os
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import skvideo.io
from IPython.display import HTML
import cv2
import skvideo.io


def showAsHTML(fpath):
    """Display a video as a Jupyter HTML widget.

    Use a relative path that is accessible via the jupyter notebook webserver.
    """
    # Add a time argument to suggest that chrome shouldn't cache the video.
    t = time.time()

    # Display images with <image>.
    for ext in '.png', '.gif', '.jpg', '.jpeg':
        if fpath.lower().endswith(ext):
            print(fpath)
            return HTML("""<image src="%s?time=%s" />""" % (fpath, t))

    # Displaly videos with <video>.
    return HTML("""
    <video width=100%% controls loop>
      <source src="%s?time=%s" type="video/mp4">
    </video>
    """ % (fpath, t))


def saveVideo(frames, fpath, **tqdmKw):
    """Save a collection of images to a video file. I've tried .mp4 extensions."""
    if tqdmKw.get('pbar', True):
        tqdmKw.setdefault('desc', os.path.basename(fpath))
        tqdmKw.setdefault('unit', 'frame')
        pbar = tqdm.tqdm_notebook
    else:
        pbar = lambda x, **kw: x
    writer = skvideo.io.FFmpegWriter(fpath)
    for frame in pbar(frames, **tqdmKw):
        writer.writeFrame(frame)
    writer.close()
    return showAsHTML(fpath)


def show(img, ax=None, title=None, clearTicks=True):
    """Display an image without x/y ticks."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    if clearTicks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None: ax.set_title(title)
    return ax.figure, ax


def drawShape(img, pts, color=(0, 255, 0), alpha=1, beta=.3):
    """Fill a polygon on an image."""
    bright = np.copy(img)
    cv2.fillPoly(bright, np.int_([pts]), color)
    return cv2.addWeighted(img, alpha, bright, beta, 0)


def fig2img(fig):
    """Render a Matplotlib figure to an image; good for simple video-making."""
    # stackoverflow.com/questions/35355930
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    canvas = FigureCanvas(fig)
    ax = fig.gca()
    canvas.draw()       # draw the canvas, cache the renderer
    width, _ = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(-1, width, 3)
    return image


def isInteractive():
    """Are we in a notebook?"""
    import __main__ as main
    return not hasattr(main, '__file__')


def bk():
    """Set a breakpoint; for use in Jupyter."""
    from IPython.core.debugger import set_trace
    set_trace()


def drawLine(x, y, canvas, **kwargs):
    kwargs.setdefault('isClosed', False)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return cv2.polylines(
        canvas, np.int32([np.stack([x, y]).T]),
        **kwargs
    )


def loadFrames(videoPrefices=('project', 'challenge', 'harder_challenge'), maxFrames=None, pbar=False):
    allFrames = {}
    for videoPrefix in videoPrefices:
        fpath = '%s_video.mp4' % videoPrefix
        reader = skvideo.io.FFmpegReader(fpath)
        frames = []
        actualMaxFrames = reader.inputframenum if maxFrames is None else maxFrames
        if pbar:
            bar = tqdm.tqdm_notebook(
                total=actualMaxFrames,
                desc='load %s' % videoPrefix,
            )
            update = bar.update
        else:
            update = lambda : None

        for frame in reader.nextFrame():
            if len(frames) == actualMaxFrames:
                break
            update()
            frames.append(frame)
        allFrames[videoPrefix] = frames
    return allFrames


def transformVideo(filePathOrFrames, outPath, transformFunction, giveFrameNum=False, **tqdmKw):

    if isinstance(filePathOrFrames, str):
        reader = skvideo.io.FFmpegReader(filePathOrFrames)
        frameSource = reader.nextFrame()
        total = reader.inputframenum
        tqdmKw['total'] = total
    else:
        frameSource = filePathOrFrames

        if 'total' not in tqdmKw:
            total = len(frameSource)
            tqdmKw['total'] = total

    if giveFrameNum:
        f = lambda frame, frameNum: transformFunction(frame, frameNum=frameNum)
    else:
        f = lambda frame, frameNum: transformFunction(frame)
    return saveVideo((f(frame, frameNum) for (frameNum, frame) in enumerate(frameSource)), outPath, **tqdmKw)


class ShowOpClip:

    def __init__(self, frames):
        self.frames = frames

    def __call__(self, op):
        pipeline = op.getByKind(cf.Pipeline, index=0)
        
        from os import system
        import time
        system('mkdir -p doc/images/')
        fpath = 'doc/images/%s-%s.gif' % (type(pipeline).__name__, type(op).__name__)
        
        def f(frame):
            pipeline(frame)
            return op.showMembersFast(show=False)[0].X
        return laneFindingPipeline.utils.transformVideo(self.frames, fpath, f, pbar=False)

