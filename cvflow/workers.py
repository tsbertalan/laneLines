import cv2, tqdm

from cvflow.baseOps import *
from cvflow.misc import isInteractive

class UndistortTransformer(object):
    """Remove barrel distortion given checkerboard calibration images."""
    
    def __init__(self, nx=9, ny=6, pbar=False):
        self.nx = nx
        self.ny = ny
        self.pbar = pbar
        
        self.singleObjP = np.zeros((nx*ny, 3), np.float32)
        self.singleObjP[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        self.imgp = []
        
    def fitImg(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.imageShape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        if ret:
            self.imgp.append(corners)
            
    def fit(self, imgs=None):
        """
        Parameters
        ----------
        imgs : iterable of str or ndarray
            Either paths to image files or the images loaded with cv2.imread
        """
        if imgs is None:
            from glob import glob
            imgs = glob('camera_cal/*.jpg')

        if self.pbar:
            if isInteractive: bar = tqdm.tqdm_notebook
            else: bar = tqdm.tqdm
        else:
            bar = lambda x, **kw: x
        for img in bar(imgs, unit='frame', desc='cal. undistort'):
            self.fitImg(img)
        self.calcParams()
        
    def calcParams(self):
        objp = [self.singleObjP] * len(self.imgp)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objp, self.imgp, self.imageShape, None, None
        )
        
    def __call__(self, img):
        """
        Parameters
        ----------
        img : ndarray

        Returns
        -------
        out : ndarray
            The undistorted image.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


class PerspectiveTransformer(object):
    """Warp a trapezoid to fill a rectangle."""
    
    def __init__(self, shape=(1280, 720), horizonRadius=.1, horizonLevel=.65, hoodPixels=30, d=150):
        """
        Parameters
        ----------
        shape : tuple of ints, optional
            width and height of the input image
        hoizonRadius : float, optional
            Fraction of the image width occupied by the top of the trapezoid.
        horizonLevel : float, optional
            Fraction of the height down from the top at which the trapezoid height is located.
        hoodPixels : int, optional
            How many pixels to crop from the bottom for the hood.
        d : int, optional
            Margin, in pixels, between left and right sides of image and target rectangle.
        """
        # TODO: Put the bottom of the trapezoid above the car hood.
        self.shape = shape
        self.horizonRadius = horizonRadius
        self.horizonLevel = horizonLevel
        self.hoodPixels = hoodPixels
        self.d = d
        self.setup()

    def setup(self, srcDst=None):
        if srcDst is None:
            w, h = self.shape
            d = self.d
            src = np.array(
                [
                 [w*(.5+self.horizonRadius), h*self.horizonLevel], 
                 [w*(.5-self.horizonRadius), h*self.horizonLevel], 
                 [0, h-self.hoodPixels], 
                 [w, h-self.hoodPixels],
                ]
            ).astype('float32')
            dst = np.array(
                [[w-d, 0.], [d, 0.], 
                 [d, h], [w-d, h]]
            ).astype('float32')
            srcDst = src, dst

        self.src, self.dst = src, dst = srcDst

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def callOnPoints(self, x, y, inv=False):
        if inv: M = self.Minv
        else: M = self.M
        return (
            (M[0, 0] * x + M[0, 1] * y + M[0, 2])
            /
            (M[2, 0] * x + M[2, 1] * y + M[2, 2])
            ,
            (M[1, 0] * x + M[1, 1] * y + M[1, 2])
            /
            (M[2, 0] * x + M[2, 1] * y + M[2, 2])
        )
        
    def __call__(self, img, inv=False, img_size=None):
        if img_size is None:
            assert len(img.shape) < 4
            img_size = img.shape[:2][::-1]
        if inv:
            M = self.Minv
        else:
            M = self.M
        return cv2.warpPerspective(img, M, img_size, borderMode=cv2.BORDER_CONSTANT)


def transformChessboard(img, nx=9, ny=6):
    """Use the corners of a chessboard pattern to supply src points for a perspective transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    assert ret
    src = corners[[0, nx-1, -nx, -1]].squeeze()
    dst = np.zeros_like(src)
    dx = corners[1, 0, 0] - corners[0, 0, 0]
    dy = corners[nx, 0, 1] - corners[0, 0, 1]
    l = 0+dx; r = img.shape[1] - dx
    t = 0+dx; b = img.shape[0] - dy
    dst[0, 0] = dst[2, 0] = l
    dst[0, 1] = dst[1, 1] = t
    dst[1, 0] = dst[3, 0] = r
    dst[2, 1] = dst[3, 1] = b
    transformer = PerspectiveTransformer()
    transformer.setup((src, dst))
    return transformer(img)


class CountSeekingThreshold(Boolean):
    
    def __init__(self, parent, initialThreshold=150, goalCount=None, countTol=None):
        self.threshold = initialThreshold
        self._defaultgoalCount = 10000
        self.goalCount = goalCount if goalCount is not None else self._defaultgoalCount
        self._defaultcountTol = 200
        self.countTol = countTol if countTol is not None else self._defaultcountTol
        self.iterationCounts = []
        self.addParent(parent)
        super().__init__()
        
    @cached()
    def value(self):
        channel = self.parent().value
        goalCount = self.goalCount
        countTol = self.countTol
        
        def getCount(threshold):
            mask = channel > np.ceil(threshold)
            return mask, mask.sum()
        
        threshold = self.threshold
        
        under = 0
        over = 255
        getThreshold = lambda : (over - under) / 2 + under
        niter = 0
        while True:
            mask, count = getCount(threshold)
            if (
                abs(count - goalCount) < countTol
                or over - under <= 1
            ):
                break

            if count > goalCount:
                # Too many pixels got in; threshold needs to be higher.
                under = threshold
                threshold = getThreshold()
            else: # count < goalCout
                if threshold > 254 and getCount(254)[1] > goalCount:
                    # In the special case that opening any at all is too bright, die early.
                    threshold = 255
                    mask = np.zeros_like(channel, 'bool')
                    break
                over = threshold
                threshold = getThreshold()
            niter += 1
                
        out =  max(min(int(np.ceil(threshold)), 255), 0)
        self.threshold = out
        self.iterationCounts.append(niter)
        return mask

    def __str__(self):
        paramsText = ', '.join([
            '%s=%s' % (k, getattr(self, k))
            for k in ('countTol', 'goalCount')
            if getattr(self, k) != getattr(self, '_default%s' % k)
        ])
        if len(paramsText) > 0:
            paramsText = ' (%s)' % paramsText
        return ('Thresh on "%s"' % self.parent().getSimpleName()) + paramsText


class Perspective(Op):

    def __init__(self, undistorted, **kwargs):
        self.addParent(undistorted)
        self.perspectiveTransformer = PerspectiveTransformer(**kwargs)
        super().__init__()

    @cached()
    def value(self):
        return self.perspectiveTransformer(self.parent().value)


class Undistort(Op):

    def __init__(self, camera, imgs=None, fit=True, **kwargs):
        self.addParent(camera)
        self.undistortTransformer = UndistortTransformer(**kwargs)
        super().__init__()

        if fit:
            self.undistortTransformer.fit(imgs)

    @cached()
    def value(self):
        return self.undistortTransformer(self.parent().value)


class DenseOpticalFlow(Op):
    """A stateful optical flow detector."""

    def __init__(self, parent):
        self.addParent(parent)
        self.lastParentValue = None
        self.lastFlow = None
        super().__init__()

    @cached()
    def value(self):
        x1 = self.parent().value
        x0 = self.lastParentValue
        self.lastParentValue = x1
        if x0 is None:
            out = np.ones_like(x1) * 128
        else:
            # copy parameters from https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
            flow = cv2.calcOpticalFlowFarneback(
                x0,  # prev
                x1,  # next
                self.lastFlow, # flow
                0.5, # pyr_scale
                3,   # levels
                30,  # winsize
                3,   # iterations 
                7,   # poly_n
                1.5, # poly_sigma
                cv2.OPTFLOW_USE_INITIAL_FLOW    # 
            )
            self.lastFlow = flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            out = magnitude
        return out


class SimplisticOpticalFlow(Op):
    """A stateful optical flow detector; just the difference of frames."""

    def __init__(self, parent):
        self.addParent(parent)
        self.lastParentValue = None
        super().__init__()

    @cached()
    def value(self):
        x1 = self.parent().value
        x0 = self.lastParentValue
        self.lastParentValue = x1
        if x0 is None:
            out = np.ones_like(x1) * 128
        else:
            out = x1 - x0
        return out


class RunningAverage(Op):

    def __init__(self, parent, maxlen=10):
        from collections import deque
        self.addParent(parent)
        self.storage = deque(maxlen=maxlen)
        super().__init__()

    @cached()
    def value(self):
        x = self.parent().value
        self.storage.append(x)
        return sum(self.storage) / len(self.storage)
