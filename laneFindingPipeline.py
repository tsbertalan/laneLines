import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm

from utils import show, drawShape, isInteractive


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
    
    def __init__(self, shape=(1280, 720), horizonRadius=.025, horizonLevel=.6, hoodPixels=60, d=200):
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


class ConvolutionalMarkingFinder(object):
    """Search for lane markings with a convolution."""
    
    def __init__(self, window_width=100, window_height=80, margin=100):
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
    
    def __call__(self, image, giveCentroids=False):
        """
        Parameters
        ----------
        image : ndarray
        giveCentroids : bool, optional

        Returns
        -------
        left : ndarray (2, n)
            Positions of found nonzero pixels in left lane

        right : ndarray (2, n)
            Positions of found nonzero pixels in right lane

        if giveCentroids:
            window_colCentroids : list
                Column (x) locations of the found window centroids.
        """
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        # Store the (left,right) window centroid positions per level
        window_colCentroids = []
        window_rowCentroids = []
        
        # Store the included pixel indices
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = []
        right_lane_inds = []

        # Box window
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # Gaussian window
        window = np.exp(-np.linspace(-1, 1, window_width)**2)

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

        # Add what we found for the first layer
        window_colCentroids.append((l_center,r_center))
        window_rowCentroids.append(window_height/2)
        
        def saveIndices(l_center, r_center, level):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (level+1)*window_height
            win_y_high = image.shape[0] - level*window_height
            win_xleft_low = l_center - margin
            win_xleft_high = l_center + margin
            win_xright_low = r_center - margin
            win_xright_high = r_center + margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
        saveIndices(l_center, r_center, 0)
            
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[
                int(image.shape[0]-(level+1)*window_height)
                :
                int(image.shape[0]-level*window_height)
                ,:
            ], axis=0)
            
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference 
            # is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            
            # Add what we found for that layer
            window_colCentroids.append((l_center,r_center))
            window_rowCentroids.append(window_height * level + window_height/2)
            
            saveIndices(l_center, r_center, level)
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Concatenate x and y into 2-row arrays.
        left = np.stack([leftx, lefty])
        right = np.stack([rightx, righty])

        if giveCentroids:
            return left, right, window_colCentroids
        else:
            return left, right

    def paintCentroids(self, warped):
        """Draw centroids on an image."""
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        left, right, window_centroids = self(warped, giveCentroids=True)
        leftx, lefty = left
        rightx, righty = right

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows    
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage= np.dstack((warped, warped, warped)) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

        return output
    
    def show(self, warped, **kwargs):
        """Visualize."""
        show(self.paintCentroids(warped), **kwargs)


class IncrementalMarkingFinder(object):
    """Use existing polynomial fits to guide the search for pixels in a new frame."""

    def __init__(self, margin=100):
        """
        Paramters
        ---------
        margin : int, optional
            How much to look left and right of the previous
            fit polynomial for nonzero pixels.

        """
        self.margin = margin

    def __call__(self, img, previousLaneMarkings):
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
        for previousLaneMarking in previousLaneMarkings:
            prevx = previousLaneMarking(nonzeroy)
            laneInds.append(
                (nonzerox > prevx - self.margin) & (nonzerox < prevx + self.margin)
            )

        # Extract line pixel positions.
        X = [nonzerox[laneInd] for laneInd in laneInds]
        Y = [nonzeroy[laneInd] for laneInd in laneInds]

        # Assemble 2-row point arrays.
        return [
            np.stack((x, y))
            for (x, y) in zip(X, Y)
        ]
    

class Threshold(object):
    """Apply a mixture of color or texture thresholdings to a warped image."""

    def __init__(self, s_thresh=(170, 255), sx_thresh=(20, 100)):
        """
        Parameters
        ----------
        s_thresh : tuple of int, optional
            Lower and upper bounds on S (saturation) channel.
            Picks up colorful road markings (like bright yellow centerlines).

        sx_thresh : tuple of int, optional
            Lower and upper bounds on sobel-x.
            Picks up strongly vertical edges.

        """
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def __call__(self, img, color=False):
        """
        Parameters
        ----------
        img : ndarray
        color : bool, optional
            If False, all filters will be combined into one boolean (0 or 255) array.
        """
        s_thresh = self.s_thresh
        sx_thresh = self.sx_thresh

        img = np.copy(img)

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

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

    def __init__(self, points, order=2, xm_per_pix=3.7/491.3, ym_per_pix=30/200, radiusYeval=720):
        """
        Parameters
        ----------
        points : ndarray, shape (2, n)
            Associated pixel locations

        order : int, optional
            Order for the fitting polynomial. Probably don't want to mess with this.
        """
        assert points.shape[0] == 2
        self.x, self.y = x, y = points
        # TODO: Use RANSAC to do the fitting.
        self.fit = np.polyfit(y, x, order)
        self.worldFit = np.polyfit(y*ym_per_pix, x*xm_per_pix, order)
        self.order = order
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix
        self.radiusYeval = radiusYeval

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

    def show(self, ax, plotKwargs=dict(color='green', linewidth=1), **scatterKwargs):
        """Visualize."""
        scatterKwargs.setdefault('s', 100)
        scatterKwargs.setdefault('alpha', .02)
        x, y = self.x, self.y
        order = np.argsort(y)
        y = y[order]
        x = x[order]
        ax.scatter(x, y, **scatterKwargs)

        y = np.linspace(y[0], y[-1], 1000)
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

        # Function for doing the initial marking discovery.
        self.initialDiscovery = ConvolutionalMarkingFinder()

        # Function for finding markings given a previous guesses.
        self.incrementalUpdate = IncrementalMarkingFinder()

    def preprocess(self, frame, color=False):
        # Remove barrel distortion.
        frame = self.undistort(frame)
        
        # Perspective-transform to a bird's-eye view.
        frame = self.perspective(frame)
        
        # Do a mixture of color/Sobel thresholdings.
        frame = self.threshold(frame, color=color)

        return frame
     
    def __call__(self, frame):
        frame = self.preprocess(frame)
        
        # Find the lane markings.
        if hasattr(self, 'laneMarkings'):
            # If we already have fits, go with an incremental update
            # by searching around them.
            leftPoints, rightPoints = self.incrementalUpdate(frame, self.laneMarkings)

        else:
            # Use the histogram or convolutional method
            # to find the fits on the first step.
            leftPoints, rightPoints = self.initialDiscovery(frame)

        self.laneMarkings = (
            LaneMarking(leftPoints),
            LaneMarking(rightPoints),
        )
            
        return self.laneMarkings

    def metersRightOfCenter(self, left, right, yeval=720, imgWidth=1280):
        centerline = np.mean([left(yeval), right(yeval)])
        return (imgWidth / 2. - centerline) * left.xm_per_pix

    def draw(self, frame):
        left, right = self(frame)
        laneCurve = np.zeros_like(frame)

        Ys = []
        for line in left, right:
            y = np.linspace(0, 720, 256)
            if line is right:
                y = y[::-1]
            Ys.append(y)

        points = np.hstack([
            np.stack([line(y), y])
            for (y, line) in zip(Ys, (left, right))
        ])

        cv2.fillConvexPoly(laneCurve, np.int32([points.T]), (232, 119, 34))
        laneCurve = self.perspective(laneCurve, inv=True)
        composite = cv2.addWeighted(frame, 1.0, laneCurve, 0.4, 0)
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
                thickness=4,
            )
        addInset(inset, 100, 1000)

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
        preprocessed = self.threshold(self.perspective(self.undistort(frame)), color=True)
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



