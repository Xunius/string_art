# string_art

Create string art from image, using python

This script attempts to approximate a given image by a series of strings/lines
connecting 2 points along the circumference of a circle centered at the image
center.

# General processing steps

1. Convert image to gray scale, and scale width to 400 pixels.
2. Blur slightly, invert intensity and threshold using Otsu thresholding.
3. Clip the image to within the center circle.
4. Compute Hough transformation on the thresholded image.
5. Given the number of points (N) on the circle, compute Hough parameters for all possible lines linking all `N*(N-1)/2`.
6. Compute allowed Hough parameter space based on results from step 5.
7. Pick iteratively peaks from the Hough transformation, and for each peak within
the allowed Hough parameter space, add a line with 2 end points on the circle.


Currently it is not working quite well. The output is sensitive to the type
of image and a few parameters. Any improvement or suggestion is appreciated.




# Dependencies:

Tested in python2.

Requires:

* numpy
* PIL
* skimage
* matplotlib


