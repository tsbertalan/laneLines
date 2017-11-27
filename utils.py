import os
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import skvideo.io
from IPython.display import HTML
import cv2


def showVid(fpath):
    """Display a video as a Jupyter HTML widget.

    Use a relative path that is accessible via the jupyter notebook webserver.
    """
    # Add a time argument to suggest that chrome shouldn't cache the video.
    return HTML("""
    <video width=100%% controls autoplay loop>
      <source src="%s?time=%s" type="video/mp4">
    </video>
    """ % (fpath, time.time()))


def saveVideo(frames, fpath, **tqdmKw):
    """Save a collection of images to a video file. I've tried .mp4 extensions."""
    writer = skvideo.io.FFmpegWriter(fpath)
    for frame in tqdm.tqdm_notebook(frames, desc='video: %s' % os.path.basename(fpath), unit='frame', **tqdmKw):
        writer.writeFrame(frame)
    writer.close()
    return showVid(fpath)


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
