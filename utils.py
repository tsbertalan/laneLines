import os
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import skvideo.io
from IPython.display import HTML
import cv2

def showVid(fpath):
    # Add a time argument to suggest that chrome shouldn't cache the video.
    return HTML("""
    <video width=100%% controls autoplay loop>
      <source src="%s?time=%s" type="video/mp4">
    </video>
    """ % (fpath, time.time()))

def saveVideo(frames, fpath, **tqdmKw):
    writer = skvideo.io.FFmpegWriter(fpath)
    for frame in tqdm.tqdm_notebook(frames, desc='video: %s' % os.path.basename(fpath), unit='frame', **tqdmKw):
        writer.writeFrame(frame)
    writer.close()
    return showVid(fpath)

def show(img, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None: ax.set_title(title)
    return ax.figure, ax

def drawShape(img, pts, color=(0, 255, 0), alpha=1, beta=.3):
    bright = np.copy(img)
    cv2.fillPoly(bright, np.int_([pts]), color)
    return cv2.addWeighted(img, alpha, bright, beta, 0)
