import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm

from utils import show, drawShape, isInteractive

import utils
from importlib import reload
reload(utils)


class Undistorter(object):
    """Remove barrel distortion given checkerboard calibration images."""
    
    def __init__(self, nx=9, ny=6):
        self.nx = nx
        self.ny = ny
        
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
            
    def fit(self, imgs):
        """
        Parameters
        ----------
        imgs : iterable of str or ndarray
            Either paths to image files or the images loaded with cv2.imread
        """
        if isInteractive: bar = tqdm.tqdm_notebook
        else: bar = tqdm.tqdm
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
    
    def __init__(self, shape=(1280, 720), horizonRadius=.1, horizonLevel=.65, hoodPixels=60, d=200):
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


def window_mask(width, height, img_ref, center, level):
    """Generate a boolean mask for a rectangular region."""
    href, wref = img_ref.shape[:2]
    output = np.zeros_like(img_ref)
    output[
        int(href - (level + 1) * height)
        :
        int(href - level * height),
        
        max(0, int(center - width / 2))
        :
        min(int(center + width / 2), wref)
    ] = 1
    return output

class MarkingNotFoundError(ValueError):
    pass

class MarkingFinder(object):

    # Doing this for multiple laneMarkings in each MarkingFinder
    # allows us to do some work (like convolutions) only once per frame.
    hintsx = (1280./4, 1280./4*3)

    @property
    def markings(self):
        if not hasattr(self, '_markings') or len(self._markings) == 0:
            self._markings = [LaneMarking([[hintx]*2, [0, 720]]) for hintx in self.hintsx]
        return self._markings

    @markings.setter
    def markings(self, newMarkings):
        self._markings = newMarkings

    def __getitem__(self, index):
        return self.markings[index]

    def __len__(self):
        return len(self.markings)

    def update(self, image):
        raise NotImplementedError


class ConvolutionalMarkingFinder(MarkingFinder):
    """Search for lane markings with a convolution."""
    
    def __init__(self, window_width=100, window_height=80, margin=100, windowType='gaussian'):
        """
        Parameters
        ----------
        window_width : int, optional
            Break image into (image_height / window_height) layers.

        window_width : int, optional
            Width of the kernel.

        margin : int, optional
            How much to slide left and right for searching.
        """
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.windowType = windowType

    def getSearchBoxes(self, centers, image):
        searchBoxes = []
        for center in centers:
            searchBoxes.append((
                int(max(center - self.margin, 0)),
                int(min(center + self.margin, image.shape[1])),
            ))
        return searchBoxes
    
    def update(self, image):
        """
        Parameters
        ----------
        image : ndarray

        Returns
        -------
        window_centroids : list of lists
            Column (x) locations of the found window centroids
        """

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        nlevels = int(image.shape[0] / window_height)

        # Refer to a common set of nonzero pixels.
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ## Store some things per-marking, per-level.
        # Store the window centroid positions per level.
        window_centroids = [[None] * nlevels for m in self]
        # Store the included pixel indices.
        lane_inds = [[None] * nlevels for m in self]

        # Create our window template that we will use for convolutions.
        if self.windowType == 'gaussian':
            # Gaussian window
            window = np.exp(-np.linspace(-1, 1, window_width)**2)
        else:
            # Box window.
            window = np.ones(window_width)

        # Get an initial guess for the centers.
        centers = [initialGuessMarking(image.shape[0]) for initialGuessMarking in self]

        # Iterate over horizontal slices of the image.
        for level in range(nlevels):

            # Sum over rows in this horizontal slice 
            # to get the scaled mean row.
            image_layer = np.sum(image[
                int(image.shape[0]-(level+1)*window_height)
                :
                int(image.shape[0]-level*window_height)
                ,:
            ], axis=0)

            # Convolve the window into the signal.
            conv_signal = np.convolve(window, image_layer, mode='same')

            # Find the best left centroid by using past center as a reference.
            searchBoxes = self.getSearchBoxes(centers, image)

            # Find the best centroids by using past centers as references.
            # TODO: If the bracketed portion is all-zero, consider that maybe this method has failed.
            for i in range(len(self)):
                lo, hi = searchBoxes[i]
                bracketed = conv_signal[lo:hi]
                if bracketed.any():
                    # If the convolution picked up something in this bracket,
                    # make that the new center.
                    center = np.argmax(bracketed) + lo
                else:
                    if level == 1:
                        # If this is just the second level, do nothing.
                        center = centers[i]
                    else:
                        # Otherwise, do some simple linear projection.
                        lastTwoCenters = window_centroids[i][level-2:level]
                        d = lastTwoCenters[1] - lastTwoCenters[0]
                        # If we keep finding nothing with each new level,
                        # Let the difference taper off as we go, to avoid colliding
                        # with the other lines.
                        # This is a sort of regularization in favor of vertical lines.
                        d *= .75
                        center = lastTwoCenters[1] + d

                # Save centers for the next level.
                centers[i] = center

                ## Save our results for this level.
                window_centroids[i][level] = center

                # Identify window boundaries in x and y (and right and left).
                win_y_low = image.shape[0] - (level+1)*window_height
                win_y_high = image.shape[0] - level*window_height
                win_x_low = center - window_width / 2
                win_x_high = center + window_width / 2
                
                # Identify the nonzero pixels in x and y within the window.
                good_inds = (
                    (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                    & 
                    (nonzerox >= win_x_low) & (nonzerox < win_x_high)
                ).nonzero()[0]
                
                # Insert these indices in the lists.
                lane_inds[i][level] = good_inds

        # Concatenate the arrays of indices.
        all_lane_inds = [np.concatenate(ii) for ii in lane_inds]


        # Extract line pixel positions.
        X = [nonzerox[ii] for ii in all_lane_inds]
        Y = [nonzeroy[ii] for ii in all_lane_inds]

        # Assemble 2-row point arrays.
        markingsPoints = [
            np.stack([x, y])
            for (x, y) in zip(X, Y)
        ]

        # Generate the new lane markings.
        self.markings = [
            LaneMarking(points)
            for points in markingsPoints
        ]

        self.window_centroids = window_centroids

    def paint(self, warped):
        """Draw centroids on an image."""
        assert len(warped.shape) == 2, '`warped` should be single-channel'
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        if not hasattr(self, 'window_centroids'):
            self.update(warped)
        window_centroids = self.window_centroids

        output = np.dstack((warped, warped, warped)) # making the original road pixels 3 color channels

        # Points used to draw all the left and right windows
        points = np.zeros_like(warped)

        # Go through each level and draw the windows    
        for level in range(0, len(window_centroids[0])):
            
            # Iterate over lane markings.
            for i in range(len(self)):

                # Window_mask is a function to draw window areas
                mask = window_mask(window_width, window_height, warped, window_centroids[i][level],level)

                # Do some drawing.
                if level > 0:
                    centers = [xx[level] for xx in window_centroids]
                    bounds = self.getSearchBoxes(centers, warped)

                    # Draw search box.
                    href = warped.shape[0]
                    perimeter = np.int32([np.stack([
                        [bounds[i][0], bounds[i][1], bounds[i][1], bounds[i][0]],
                        [href-level*window_height, href-level*window_height, href-(level+1)*window_height, href-(level+1)*window_height]
                    ]).T])
                    cv2.polylines(output, perimeter, isClosed=True, color=(255, 0, 0), thickness=4)

                    # Draw centroid as line.
                    perimeter = np.int32([np.stack([
                        [window_centroids[i][level]]*2,
                        [href-level*window_height, href-(level+1)*window_height,]
                    ]).T])
                    cv2.polylines(output, perimeter, isClosed=False, color=(0, 0, 255), thickness=2)

                # Add graphic points from window mask here to total pixels found 
                points[mask == 1] = 255

            # Draw the best-found windows.
            template = np.array(points, np.uint8)
            zero_channel = np.zeros_like(template) # create a zero color channel
            colored = [None]*3
            icolor = 1
            for i in range(3):
                if i == icolor:
                    colored[i] = template
                else:
                    colored[i] = zero_channel
            template = np.array(cv2.merge(colored), np.uint8) # make window pixels green
            output = cv2.addWeighted(output, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        return output
    
    def show(self, warped, **kwargs):
        """Visualize."""
        show(self.paint(warped), **kwargs)


class MarginSearchMarkingFinder(MarkingFinder):
    """Search in a margin around existing polynomial fits."""

    def __init__(self, margin=100):
        """
        Paramters
        ---------
        margin : int, optional
            How much to look left and right of the previous
            fit polynomial for nonzero pixels.

        """
        self.margin = margin

    def update(self, img):
        """Margin search method

        Parameters
        ----------
        img : ndarray
        previousLaneMarkings : list of LaneMarking
            Zero or more (probably two) LaneMarking objects.

        Returns
        -------
        points : list of ndarrays
            Zero or more coordinate arrays of shape (2, n)
        """

        # Find the points highlighted by our thresholding.
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Find indices for points with x in +/- `margin` of the old fit.
        laneInds = []
        for previousLaneMarking in self:
            prevx = previousLaneMarking(nonzeroy)
            laneInds.append(
                (nonzerox > prevx - self.margin) & (nonzerox < prevx + self.margin)
            )

        # Extract line pixel positions.
        X = [nonzerox[laneInd] for laneInd in laneInds]
        Y = [nonzeroy[laneInd] for laneInd in laneInds]

        for i in range(len(self)):
            if len(X[i]) == 0:
                # TODO: If we're going to use this method, this exception should be caught and handled by using a different method.
                raise MarkingNotFoundError('Found no points for laneMarking %d of %d.' % (i+1, len(self)))

        # Assemble 2-row point arrays.
        markingsPoints = [
            np.stack((x, y))
            for (x, y) in zip(X, Y)
        ]

        # Generate the new lane markings.
        self.markings = [
            LaneMarking(points)
            for points in markingsPoints
        ]


class Threshold(object):
    """Apply a mixture of color or texture thresholdings to a warped image."""

    def __init__(self, s_thresh=(170, 255), sx_thresh=20, dilate_kernel=(2, 6), dilationIterations=3, blurSize=(5,5)):
        """
        Parameters
        ----------
        s_thresh : tuple of int, optional
            Lower and upper bounds on S (saturation) channel.
            Picks up colorful road markings (like bright yellow centerlines).

        sx_thresh : int, optional
            -Lower and upper bounds on sobel-x.
            Picks up strongly vertical edges.

        """
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.dilate_kernel = dilate_kernel
        self.dilationIterations = dilationIterations
        self.blurSize = blurSize

    def dilateSobel(self, singleChannel):
        sx_thresh = self.sx_thresh
        dilate_kernel = self.dilate_kernel
        dilationIterations = self.dilationIterations

        sobelx = cv2.Sobel(singleChannel, cv2.CV_64F, 1, 0) # Take the derivative in x

        # Sobel mask.
        mask_neg = (sobelx < -sx_thresh).astype(np.float32)
        mask_pos = (sobelx > sx_thresh).astype(np.float32)

        mid = dilate_kernel[1] // 2
        # Dilate mask to the left.
        kernel = np.ones(dilate_kernel, np.uint8)
        kernel[:, 0:mid] = 0
        dmask_neg = cv2.dilate(mask_neg, kernel, iterations=dilationIterations) > 0.
        # Dilate mask to the right.
        kernel = np.ones(dilate_kernel, np.uint8)
        kernel[:, mid:] = 0
        dmask_pos = cv2.dilate(mask_pos, kernel, iterations=dilationIterations) > 0.
        sxbinary = (dmask_pos & dmask_neg).astype(np.uint8)
        return sxbinary

    def __call__(self, img, color=False):
        """
        Parameters
        ----------
        img : ndarray
        color : bool, optional
            If False, all filters will be combined into one boolean (0 or 255) array.
        """
        s_thresh = self.s_thresh

        img = np.copy(img)
        print('Blurring.')
        img = cv2.GaussianBlur(img, self.blurSize, 0)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Sobel x, with up/down dilation trick.
        sxbinary = self.dilateSobel(l_channel)
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        color_binary = np.dstack((
            np.zeros_like(sxbinary), # R
            sxbinary,                # G
            s_binary,                # B
        )) * 255

        if color:
            return color_binary.astype('uint8')
        else:
            cb = color_binary
            # TODO: Why is this still an 8-bit array? Would numpy pack it better if it was truly boolean?
            return np.logical_or(
                np.logical_or(
                    cb[:, :, 0], cb[:, :, 1]),
                cb[:, :, 2]
            ).astype('uint8') * 255


class LaneMarking(object):
    """Convenince class for storing polynomial fit and pixels for a lane marking."""
    # TODO: Try using a histogram of y values to classify broken and solid markings.

    def __init__(self, points, order=2, xm_per_pix=3.7/491.3, ym_per_pix=30/300, radiusYeval=720, imageSize=(720, 1280)):
        """
        Parameters
        ----------
        points : ndarray, shape (2, n)
            Associated pixel locations

        order : int, optional
            Order for the fitting polynomial. Probably don't want to mess with this.
        """
        points = np.asarray(points)
        assert points.shape[0] == 2
        self.x, self.y = x, y = points
        # TODO: Use RANSAC to do the fitting.
        # Construct the fits by hand if a vertical line is given,
        # to avoid RankWarning from numpy.
        if len(set(x)) == 1:
            self.fit = np.zeros((order+1,))
            self.fit[-1] = x[0]
            self.worldFit = np.zeros((order+1,))
            self.worldFit[-1] = x[0] * xm_per_pix
        else:
            self.fit = np.polyfit(y, x, order)
            self.worldFit = np.polyfit(y * ym_per_pix, x * xm_per_pix, order)
        self.order = order
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        self.radiusYeval = radiusYeval
        self.imageSize = imageSize

    def __call__(self, y=None):
        """Calculate y (row) as a function of x (column).

        Parameters
        ----------
        y : ndarray, optional
            If not given, the data we used to generate the fit will be reused.
        """
        if y is None:
            y = self.y
        # Even though there will be duplicate values in y,
        # this default vectorized version is about 50% faster
        # than a dict-backed memoized version.
        # This might be different with thicker lane markings
        # (more y values to map to the same x).
        return np.polyval(self.fit, y)

    def show(self, ax, plotKwargs=dict(color='magenta', linewidth=4), **scatterKwargs):
        """Visualize."""
        scatterKwargs.setdefault('s', 100)
        scatterKwargs.setdefault('alpha', .02)
        x, y = self.x, self.y
        order = np.argsort(y)
        y = y[order]
        x = x[order]
        ax.scatter(x, y, **scatterKwargs)

        y = np.linspace(0, self.imageSize[0], 1000)
        x = self(y)
        ax.plot(x, y, **plotKwargs)

        return ax.figure, ax

    @property
    def radius(self):
        yeval = self.radiusYeval * self.ym_per_pix
        fit = np.poly1d(self.worldFit)
        xp = np.polyder(fit,  m=1)
        xpp = np.polyder(fit, m=2)
        num = (1 + xp ** 2)(yeval)
        den = abs(xpp(yeval))
        return num ** (3./2) / den


class LaneFinder(object):
    """Stateful lane-marking finder for hood camera video."""
    
    def __init__(self, 
        undistort=None,
        ):

        # SET ALL CALLABLE ATTRIBUTES.
        
        # Function for removing barrel distortion.
        if undistort is None:
            undistort = Undistorter()
            import glob
            undistort.fit(glob.glob('camera_cal/*.jpg'))
        self.undistort = undistort

        # Function for applying various color/Sobel thresholds.
        self.threshold = Threshold()
        
        # Function for transforming perspective.
        self.perspective = PerspectiveTransformer()

        # Function for marking discovery.
        self.markingFinder = ConvolutionalMarkingFinder()

    def preprocess(self, frame, color=False):
        # Remove barrel distortion.
        frame = self.undistort(frame)
        
        # Perspective-transform to a bird's-eye view.
        frame = self.perspective(frame)
        
        # Do a mixture of color/Sobel thresholdings.
        frame = self.threshold(frame, color=color)

        return frame
     
    def __call__(self, frame):
        # Do our preprocessing.
        frame = self.preprocess(frame)
        
        # Find the lane markings.
        self.markingFinder.update(frame)
        self.laneMarkings = self.markingFinder.markings

        return self.laneMarkings

    def metersRightOfCenter(self, left, right, yeval=720, imgWidth=1280):
        centerline = np.mean([left(yeval), right(yeval)])
        return (imgWidth / 2. - centerline) * left.xm_per_pix

    def draw(self, frame, showTrapezoid=True, showThresholds=True, insetThresholds=True, showLane=True):
        left, right = self(frame)
        
        composite = np.copy(frame)

        # Draw the lane curve.
        Ys = []
        for line in left, right:
            y = np.linspace(0, 720, 256)
            if line is right:
                y = y[::-1]
            Ys.append(y)

        if showLane:
            points = np.hstack([
                np.stack([line(y), y])
                for (y, line) in zip(Ys, (left, right))
            ])
            laneCurve = np.zeros_like(frame)
            cv2.fillConvexPoly(laneCurve, np.int32([points.T]), (232, 119, 34))
            for (y, line) in zip(Ys, (left, right)):
                cv2.polylines(
                    laneCurve, np.int32([np.stack([line(y), y]).T]),
                    isClosed=False, color=(255, 0, 0),
                    thickness=32,
                )
            laneCurve = self.perspective(laneCurve, inv=True)
            composite = cv2.addWeighted(composite, 1.0, laneCurve, 0.4, 0)

        if showTrapezoid:
            cv2.polylines(
                composite, np.int32([self.perspective.src]), 
                isClosed=True, color=(255, 105, 180),
                thickness=1,
            )

        def addInset(img, r, c, f=.2):
            newsize = int(img.shape[1]*f), int(img.shape[0]*f)
            small = cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)
            composite[r:r+small.shape[0], c:c+small.shape[1], :] = small

        inset = self.preprocess(frame, color=True)
        for (y, line) in zip(Ys, (left, right)):
            cv2.polylines(
                inset, np.int32([np.stack([line(y), y]).T]),
                isClosed=False, color=(255, 0, 0),
                thickness=12,
            )
        centroids = self.markingFinder.paint(self.preprocess(frame))
        inset = cv2.addWeighted(inset, 1.0, centroids, 0.5, 0)

        if insetThresholds:
            addInset(inset, 100, 1000)

        if showThresholds:
            composite = cv2.addWeighted(composite, 1.0, self.perspective(inset, inv=True), 0.8, 0)

        y = 50
        for text in [
            'left radius: %.5g [m]' % left.radius,
            'right radius: %.5g [m]' % right.radius,
            'offset from centerline: %.3g [m]' % self.metersRightOfCenter(left, right),
        ]:
            cv2.putText(
                composite, 
                text,
                (10, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA,
            )
            y += 50

        return composite

    def show(self, frame, axes=None):
        """Visualize.

        About 6x slower than function itself, fyi.
        """
        if axes is None:
            fig, axes = plt.subplots(nrows=2)

        if len(axes) > 1:
            #Show the original frame.
            show(frame, ax=axes[0])

        ax = axes[-1]

        # Show the preprocessed image on the bottom.
        perspectived = self.perspective(self.undistort(frame))
        preprocessed = self.threshold(perspectived, color=True)
        if isinstance(self.markingFinder, ConvolutionalMarkingFinder):
            warped = self.threshold(perspectived)
            preprocessed = cv2.addWeighted(preprocessed, 1.0, self.markingFinder.paint(warped), 0.5, 0)
        ax.imshow(preprocessed)

        # Plot the lane markings and (faintly) their associated pixels.
        for laneMarking in self(frame):
            laneMarking.show(ax)

        # Clean up the figure.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        ax.figure.tight_layout()

        return ax.figure, axes



