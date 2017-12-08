## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goal of this project was to identify curved lane markings from hood- or dash-mounted camera view. Broadly, the complete pipeline used three steps:
1. Correct for barrel distortion and use a four-point perspective transformation to produce a top-down view.
2. Use a combination of thresholding, image derivatives, and other OpenCV effects to extract only the pixels associated with the two lane markings.
3. Produce polynomial fit lines to the found pixels, both in image space, and in world space. Use the latter to estimate the radius of curvature of the lane, and the car's deviation from center (in meters).

A writeup of these results is visible in "Advanced Lane Finding.ipynb", or rendered to PDF in "report.pdf".
