import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import tqdm
import skvideo.io

import cvflow as cf

from utils import show, drawShape, isInteractive
import utils
import smoothing


def windowMask(width, height, img_ref, center, level):
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
    hintsx = (1280. * .25 , 1280. * .75)

    def getHintedLaneMarking(self, i):
        return LaneMarking([[self.hintsx[i]]*2, [0, 720]])

    @property
    def markings(self):
        if not hasattr(self, '_markings') or len(self._markings) == 0:
            self._markings = [self.getHintedLaneMarking(i) for i in range(len(self.hintsx))]
        return self._markings

    @markings.setter
    def markings(self, newMarkings):
        self._markings = newMarkings

    def __getitem__(self, index):
        return self.markings[index]

    def __len__(self):
        return len(self.markings)

    def update(self, image, _recursionDepth=0):
        raise NotImplementedError

    def evaluateFitQuality(self, lowPointsThresh=1, radiusRatioThresh=500):
        self.failedLaneMarkings = []

        ## Check single-marking quality indicators.
        # Some points were actually found?
        acceptables = [True] * len(self)
        for i, laneMarking in enumerate(self):
            
            if len(laneMarking.x) <= lowPointsThresh:
                acceptables[i] = False

        ## Check paired quality indicators:
        # Radii aren't drastically different?
        assert len(self) == 2
        left, right = self
        rads = [left.radius, right.radius]
        rads.sort()
        radRat = rads[1] / rads[0] 
        if radRat > radiusRatioThresh:
            acceptables = [False] * 2

        # The two fits aren't actually the same?
        a = left.worldFit
        b = right.worldFit
        assert len(a) == len(b)
        # Different thresholds are appropriate for different coeffients.
        thresholds = [10**-(k+1) for k in range(len(a))][::-1]
        if not (np.abs(a - b) > thresholds).any():
            acceptables = [False] * 2

        # Assign quality scores.
        qualityFactors = 10, 1, 1; norm = sum(qualityFactors)
        for i, laneMarking in enumerate(self):
            mse = laneMarking.mse
            n = len(laneMarking.x)
            qualityVec = (
                n / 9001.,
                100. / (mse if (mse != 0 and n > 2)  else np.inf),
                3.0 / radRat,
            )

            laneMarking.quality = sum([q*fac/norm for (q, fac) in zip(qualityVec, qualityFactors)]) / len(qualityVec)
            laneMarking.qualityVec = qualityVec

        for i, acceptable in enumerate(acceptables):
            if not acceptable:
                self.failedLaneMarkings.append(self.markings[i])
                self.markings[i] = self.getHintedLaneMarking(i)

        # Report to the caller whether all laneMarkings passed inspection.
        return np.all(acceptables)

    def postUpdateQualityCheck(self, image, _recursionDepth, maxRecursion=1):
        # Check some quality metrics on the found lines.
        # If they're found lacking, the checker will replace them
        # with the default guesses. We can then try the update again.
        if not self.evaluateFitQuality():

            if _recursionDepth < maxRecursion:
                self.update(image, _recursionDepth=_recursionDepth+1)


class ConvolutionalMarkingFinder(MarkingFinder):
    """Search for lane markings with a convolution."""
    
    def __init__(self, 
        windowWidth=50, windowHeight=40, 
        searchMargin=(75, 120), 
        windowType='gaussian', gaussianRadius=1.5, 
        ):
        """
        Parameters
        ----------
        windowWidth : int, optional
            Width of the kernel.

        windowHeight : int, optional
            Break image into (image_height / windowHeight) layers.

        searchMargin : int, optional
            How much to slide left and right for searching.

        windowType : str, optional
            Either 'gaussian' or 'box'. What kind of convolution kernel to use.

        gaussianRadius : float, optional
            If windowType=='gaussian', how wide the Gaussian should be. It's always
            going to be windowWidth samples wide, but those will range from
            -gaussianRadius to +gaussianRadius, evaluating exp(-x^2)
        """
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.searchMargin = searchMargin
        self.windowType = windowType
        self.gaussianRadius = gaussianRadius

    def _getSearchBoxes(self, centers, image, level, nlevels):
        """Reduce the whole image width into restricted search boxes given the previous centers."""
        searchBoxes = []
        searchMargin = self.searchMargin
        if hasattr(searchMargin, '__getitem__'):
            searchMargin = np.linspace(searchMargin[0], searchMargin[1], nlevels)[level]

        for center in centers:
            searchBoxes.append((
                int(max(center - searchMargin, 0)),
                int(min(center + searchMargin, image.shape[1])),
            ))
        return searchBoxes
    
    def update(self, image, _recursionDepth=0):
        """Find lane markings in preprocessed image.

        Parameters
        ----------
        image : ndarray

        (Ignore _recursionDepth)

        Sets attributes
        ---------------
        self.markings : list
            The found LaneMarking objects
        self.windowCentroids : list of lists
            Column (x) locations of the found window centroids
        """
        windowWidth = self.windowWidth
        windowHeight = self.windowHeight
        gaussianRadius = self.gaussianRadius
        nlevels = int(image.shape[0] / windowHeight)

        # Refer to a common set of nonzero pixels.
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ## Store some things per-marking, per-level.
        # Store the window centroid positions per level.
        windowCentroids = [[None] * nlevels for m in self]
        # Store the included pixel indices.
        lane_inds = [[None] * nlevels for m in self]

        # Create our window template that we will use for convolutions.
        if self.windowType == 'gaussian':
            # Gaussian window
            window = np.exp(-np.linspace(-gaussianRadius, gaussianRadius, windowWidth)**2)
        else:
            # Box window.
            window = np.ones(windowWidth)

        # Get an initial guess for the centers.
        centers = [initialGuessMarking(image.shape[0]) for initialGuessMarking in self]

        # Iterate over horizontal slices of the image.
        for level in range(nlevels):

            # Sum over rows in this horizontal slice 
            # to get the scaled mean row.
            image_layer = np.sum(image[
                int(image.shape[0]-(level+1)*windowHeight)
                :
                int(image.shape[0]-level*windowHeight)
                ,:
            ], axis=0)

            # Convolve the window into the signal.
            conv_signal = np.convolve(window, image_layer, mode='same')

            # Find the best left centroid by using past center as a reference.
            searchBoxes = self._getSearchBoxes(centers, image, level, nlevels)

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
                    # Do nothing.
                    center = centers[i]
                    
                # Save centers for the next level.
                centers[i] = center

                ## Save our results for this level.
                windowCentroids[i][level] = center

                # Identify window boundaries in x and y (and right and left).
                win_y_low = image.shape[0] - (level+1)*windowHeight
                win_y_high = image.shape[0] - level*windowHeight
                win_x_low = center - windowWidth / 2
                win_x_high = center + windowWidth / 2
                
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

        # If the location arrays are somehow totally empty, revert to default vertical lines.
        for i in range(len(self)):
            if len(X[i]) == 0:
                assert len(Y[i]) == 0
                X[i] = np.array((self.hintsx[i],), X[i].dtype)
                Y[i] = np.zeros((1,), Y[i].dtype)

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

        self.windowCentroids = windowCentroids

        self.postUpdateQualityCheck(image, _recursionDepth)

    def paint(self, warped):
        """Draw centroids on an image."""
        assert len(warped.shape) == 2, '`warped` should be single-channel'
        windowWidth = self.windowWidth
        windowHeight = self.windowHeight
        
        if not hasattr(self, 'windowCentroids'):
            self.update(warped)
        windowCentroids = self.windowCentroids

        output = np.dstack((warped, warped, warped)) # making the original road pixels 3 color channels

        # Points used to draw all the left and right windows
        points = np.zeros_like(warped)

        # Go through each level and draw the windows    
        nlevels = int(warped.shape[0] / windowHeight)
        for level in range(0, nlevels):

            # Iterate over lane markings.
            for i in range(len(self)):

                # Window_mask is a function to draw window areas
                mask = windowMask(windowWidth, windowHeight, warped, windowCentroids[i][level],level)

                # Do some drawing.
                if level > 0:
                    centers = [xx[level] for xx in windowCentroids]
                    bounds = self._getSearchBoxes(centers, warped, level, nlevels)

                    # Draw search box.
                    href = warped.shape[0]
                    perimeter = np.int32([np.stack([
                        [bounds[i][0], bounds[i][1], bounds[i][1], bounds[i][0]],
                        [href-level*windowHeight, href-level*windowHeight, href-(level+1)*windowHeight, href-(level+1)*windowHeight]
                    ]).T])
                    cv2.polylines(output, perimeter, isClosed=True, color=(255, 0, 0), thickness=4)

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
            output = cv2.addWeighted(output.astype('uint8'), 1, template, 0.5, 0.0) # overlay the orignal road image with window results

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
        """Find lane markings in preprocessed image.

        Parameters
        ----------
        image : ndarray

        (Ignore _recursionDepth)

        Sets attributes
        ---------------
        self.markings : list
            The found LaneMarking objects
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


class LaneMarking(object):
    """Convenince class for storing polynomial fit and pixels for a lane marking."""
    # TODO: Try using a histogram of y values to classify broken and solid markings.

    def __init__(self, points=None, 
        order=2,
        ransac=False, lamb=.5, 
        smoothers=(lambda x: x, lambda x: x)
        ):
        """
        Parameters
        ----------
        points : ndarray, shape (2, n)
            Associated pixel locations
        order : int, optional
            Order for the fitting polynomial. Probably don't want to mess with this.
        ransac : bool, optional
            Whether to use RANSAC or normal polynomial regression.
        lamb : float, optional
            L2 regularization coefficient.
        smoothers : tuple of two callables
            Functions that take in a coefficient vector and return a vector 
            of the same size, perhaps with some stateful smoothing applied.

        """

        # Default case for no-argument construction
        self.imageSize = (720, 1280)
        if points is None:
            self.initialized = False
            points = [[self.imageSize[1]/2]*10, np.linspace(0, self.imageSize[0], 10)]
        else:
            self.initialized = True
        
        points = np.asarray(points)
        assert points.shape[0] == 2
        self.x, self.y = x, y = points
        self.order = order
        self.xm_per_pix = 3.6576/628.33
        self.ym_per_pix = 18.288/602
        self.radiusYeval = 720
        self.smoothers = smoothers
        self.quality = float(len(x))
        self.qualityVec = [None]*3

        # TODO: Use RANSAC to do the fitting.
        # Construct the fits by hand if a vertical line is given,
        # to avoid RankWarning from numpy.
        if len(set(x)) == 1:
            self.fit = np.zeros((order+1,))
            self.fit[-1] = x[0]
            self.worldFit = np.zeros((order+1,))
            self.worldFit[-1] = x[0] * self.xm_per_pix
            self.mse = 0
        else:
            self.fit = self.regressPoly(y, x, order, ransac=ransac, lamb=lamb)
            self.worldFit = self.regressPoly(y * self.ym_per_pix, x * self.xm_per_pix, order, ransac=ransac, lamb=lamb)
            self.mse = np.mean((self() - x)**2)

    @staticmethod
    def regressPoly(x, y, order, ransac=False, lamb=0.5):
        """Regress a polynomial for y = f(x).

        Parameters
        ----------
        x : iterable, length n
            Independent/predictor variable
        y : iterable, length n
            Dependent/response variable
        order : int
            Polynomial order
        ransac : bool, optional
            Whether to use RANSAC robust regression
        lamb : float, optional
            L2 regularization coefficient

        Returns
        -------
        fit : ndarray, length order+1
            The monomial coefficients in decreasing-power order.

        """
        # scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html
        # scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
        if lamb == 0:
            if not ransac:
                return np.polyfit(x, y , order)
            else:
                estimator = RANSACRegressor(random_state=42, min_samples=.8, max_trials=300)
                model = make_pipeline(PolynomialFeatures(order), estimator)
                model.fit(x.reshape(x.size, 1), y)
                return model._final_estimator.estimator_.coef_[::-1]
        else:
            A = np.stack([np.asarray(x).ravel()**k for k in range(order+1)[::-1]]).T
            n_col = A.shape[1]
            fit, residuals, rank, s = np.linalg.lstsq(
                A.T.dot(A) + lamb * np.identity(n_col), 
                A.T.dot(y)
            )
            return fit

    def __call__(self, y=None, worldCoordinates=False):
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
        if worldCoordinates:
            fit = self.worldFit
        else:
            fit = self.fit
        return np.polyval(fit, y)

    def show(self, ax, plotKwargs=dict(color='magenta', linewidth=4, alpha=.5), **scatterKwargs):
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
        """Radius of curvature near the camera, in meters."""
        yeval = self.radiusYeval * self.ym_per_pix
        fit = np.poly1d(self.worldFit)
        xp = np.polyder(fit,  m=1)
        xpp = np.polyder(fit, m=2)
        num = (1 + xp ** 2)(yeval)
        den = abs(xpp(yeval))
        if den == 0:
            return np.inf
        return num ** (3./2) / den

    def update(self, otherMarking):
        weight = {}
        if isinstance(self.smoothers[0], smoothing.BoxSmoother):
            weight = {'weight': otherMarking.quality}
        self.fit = self.smoothers[0](otherMarking.fit, **weight)
        self.worldFit = self.smoothers[1](otherMarking.worldFit, **weight)


class LaneFinder(object):
    """Stateful lane-marking finder for hood camera video."""
    
    def __init__(self, 
        colorFilter=cf.FullPipeline(),
        markingFinder=ConvolutionalMarkingFinder(),
        Smoother=smoothing.WeightedSmoother,
        ):
        """
        Parameters
        ----------
        colorFilter : callable, optional
            Should ingest a three-channel image array 
            and return a single-channel boolean mask on lane-marking pixels.
        markingFinder : MarkingFinder, optional
        Smoother : type, optional
            Subclass of smoothing.Smoother that will be instantiated without
            arguments when creating LaneMarking objects.

        """

        # SET ALL CALLABLE ATTRIBUTES.
        
        # Function for applying various color/Sobel thresholds.
        self.colorFilter = colorFilter
        self.perspective = self.colorFilter.getMembersByType(cf.workers.Perspective)[0].perspectiveTransformer
        #self.undistort   = self.colorFilter.getMembersByType(cf.workers.Undistort  ).undistortTransformer
        
        # Function for marking discovery.
        self.markingFinder = markingFinder

        # Smoother for found fits.
        self.Smoother = Smoother

        # Locally stored markings for smoothing.
        self.laneMarkings = (
            LaneMarking(smoothers=(Smoother(), Smoother())), 
            LaneMarking(smoothers=(Smoother(), Smoother())),
        )

    def preprocess(self, frame, color=False):
        """Convert a camera image to a perspective-transformed mask on lane-marking pixels."""
        return self.colorFilter(frame, color=color)

    def update(self, preprocessed):
        """Update self.laneMarkings with preprocessed marking pixels mask."""
        self.markingFinder.update(preprocessed)
        newLaneMarkings = self.markingFinder.markings
        for oldLaneMarking, newLaneMarking in zip(self.laneMarkings, newLaneMarkings):
            oldLaneMarking.update(newLaneMarking)
        return self.laneMarkings
     
    def __call__(self, frame):
        """Update and return self.laneMarkings with camera frame."""
        # Do our preprocessing.
        preprocessed = self.preprocess(frame)
        
        # Find the lane markings.
        self.update(preprocessed)

        # Return the smoothed lane markings.
        return self.laneMarkings

    def metersRightOfCenter(self, left, right, yeval=720, imgWidth=1280):
        """Deviation of camera center from lane center, in meters."""
        centerline = np.mean([left(yeval), right(yeval)])
        return (imgWidth / 2. - centerline) * left.xm_per_pix

    def draw(self, frame, 
        call=True, showTrapezoid=True, showThresholds=True, 
        insetBEV=True, showLane=True, showCurves=False, showCentroids=True
        ):
        """Visualize.

        Parameters
        ----------
        frame : ndarray
            The camera image
        call : bool, optional
            Whether to do self(frame), updating the laneMarkings
        showTrapezoid : bool, optional
            Whether to show the trapezoid representing the perspective transform
        showThresholds : bool, optional
            Whether to show the pixels identified by the preprocessing
        insetBEV : bool, optional
            Whether to show the birds-eye-view inset
        showLane : bool, optional
            Whether to show the foudn lane-shape overlay
        showCurves : bool, optional
            Whether to draw the marking polynomials as thick curves
        showCentroids : bool, optional
            Whether to show the boxes identified by the MarkingFinder

        Returns
        -------
        composite : ndarray
            Color image with all the drawn features.
        """
        if call:
            self(frame)
        preprocessed = self.colorFilter.output.value
        preprocessed_color = self.colorFilter.colorOutput.value
        left, right = self.laneMarkings
        
        composite = np.copy(frame)

        # Draw the lane curve.
        # Draw one line up and the other down,
        # so the polygon between will fill correctly.
        Ys = []
        for line in left, right:
            y = np.linspace(0, 720, 256)
            if line is right:
                y = y[::-1]
            Ys.append(y)

        # Add an inset plot showing the top-down view.
        def addInset(img, r, c, f=.2):
            newsize = int(img.shape[1]*f), int(img.shape[0]*f)
            small = cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)
            composite[r:r+small.shape[0], c:c+small.shape[1], :] = small
        inset = np.zeros_like(frame)

        # Add various things to the inset.
        if showLane:
            points = np.hstack([
                np.stack([line(y), y])
                for (y, line) in zip(Ys, (left, right))
            ])
            laneCurve = np.zeros_like(frame)
            cv2.fillConvexPoly(laneCurve, np.int32([points.T]), (232, 119, 34))
            inset = cv2.addWeighted(inset, 1.0, laneCurve, 0.4, 0)

        if showTrapezoid:
            x = self.perspective.dst[:, 0]
            y = self.perspective.dst[:, 1]
            utils.drawLine(x, y, inset, color=(255, 105, 180), isClosed=True, thickness=3)

        if showThresholds:
            inset = cv2.addWeighted(inset, 1.0, preprocessed_color, 1.0, 0)

        if showCurves:
            y = Ys[0]
            for line in (left, right):
                utils.drawLine(
                    line(y), y, inset,
                    color=(255, 0, 0),
                    thickness=12,
                )
            for line in self.markingFinder.failedLaneMarkings:
                utils.drawLine(line(y), y, inset, color=(128, 128, 128), thickness=12)

        if showCentroids:
            centroids = self.markingFinder.paint(preprocessed)
            inset = cv2.addWeighted(inset, 1.0, centroids, 0.5, 0)

        # Warp the inset down onto the main view.
        composite = cv2.addWeighted(composite, 1.0, self.perspective(inset, inv=True), 0.8, 0)

        if insetBEV:
            if not hasattr(self, 'carDecal'):
                self.carDecal = cv2.imread('carOverlay.png')
            inset = cv2.addWeighted(self.carDecal, 1.0, inset, 1.0, 0)
            addInset(inset, 100, 1000)

        # Add a text overlay.
        y = 50
        alphaStrings = [
            'a(%d history) = [%s]' % (
                len(self.laneMarkings[i].smoothers[0].history),
                ', '.join([
                    '%.4g' % a
                    for a in self.laneMarkings[i].worldFit
                ])
            )
            for i in range(len(self.laneMarkings))
        ]
        texts = []
        for i in range(2):
            n = ('left', 'right')[i]
            texts.append('%s radius: %.5g [m]' % (n, self.laneMarkings[i].radius))
            texts.append(alphaStrings[0])
            if isinstance(self.laneMarkings[0].smoothers[0], smoothing.WeightedSmoother):
                texts.append('last window weight %s' % self.laneMarkings[i].smoothers[0].window[-1])
        texts.append('offset from centerline: %.3g [m]' % self.metersRightOfCenter(left, right))
        for text in texts:
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

    def drawSteps(self, frame, **drawKwargs):
        """Visualize the preprocessing steps.
        
        Parameters
        ----------
        frame : ndarray
            The camera image
        **drawKwargs
            Other keyword arguments are passed on to draw()

        """
        drawing = self.draw(frame, **drawKwargs)
        self.colorFilter.showMembersFast(show=False, recurse=False)
        plotter = self.colorFilter.addExtraPlot(drawing)
        return plotter.X

    def process(self,
        filePathOrFrames, outFilePath, frame0=0,
        showSteps=False, maxFrames=None, drawFrameNum=True, 
        tqdmKw={'pbar': True}, **drawKwargs
        ):
        """Process all frames in a video.

        Parameters
        ----------
        filePathOrFrames : str or list
            Either a list of ndarray camera frames, or a path to a video file
        outFilePath : str
            Path to output video or GIF file
        frame0 : int, optional
            Frame number of the first frame, for drawFrameNum
        showSteps : bool, optional
            Whether to use drawSteps instead of draw.
        maxFrames : int, optional
            Limit on number of frames to process
        drawFrameNum : bool, optional
            Whether to draw the frame number in the corner of the output.
        tqdmKw : dict, optional
            Passed on as keyword arguments to the video saver
        **drawKwargs
            Passed on to the drawing method.

        Returns
        -------
        videoHtml
            An object that implements _repr_html_, for easy display in Jupyter

        """

        # Make a frame reader object or just use the provided frames.
        if isinstance(filePathOrFrames, str):
            reader = skvideo.io.FFmpegReader(filePathOrFrames)
            frameSource = reader.nextFrame()
            total = reader.inputframenum
        else:
            frameSource = filePathOrFrames
            total = len(frameSource)

        # Cap the frames output.
        total = min(total, maxFrames) if maxFrames is not None else total
        tqdmKw['total'] = total

        # Draw the frame and write the frame number on it.
        def yieldFrames():
            for frameNum, frame in enumerate(frameSource):
                if frameNum == total:
                    break
                else:
                    if showSteps:
                        response = self.drawSteps(frame, **drawKwargs)
                    else:
                        response = self.draw(frame, **drawKwargs)
                    cv2.putText(
                        response,
                        'Frame %d' % (frameNum + frame0,), 
                        (response.shape[1] - 400, response.shape[0] - 64),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        thickness=2,
                        fontScale=2,
                        color=(200, 200, 200),
                        lineType=cv2.LINE_AA,
                    )
                    yield response

        videoHtml = utils.saveVideo(
            yieldFrames(),
            outFilePath,
            **tqdmKw
            )
        return videoHtml

