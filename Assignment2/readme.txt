COMP3317 Computer Vision
Assignment 2 - Corner detection

1. Goal
 • colour-to-grayscale image conversion
 • corner detection

2. Implemented Features
（1）rgb2gray()
 • Get r, g, b from the colour image.
 • Use the formula “Y = 0.299*r + 0.587*g + 0.144*b” for the Y-channel of the YIQ model in performing the colour-to-grayscale image conversion.

（2）smooth1D()
 • Using n = int(sigma * (2*np.log (1000))**0.5) to compute a proper filter size for a Gaussian filter based on its sigma value.
 • Using x = np.arange(-n, n+1) to form a kernel with proper size.
 • Construct and truncate a proper 1D Gaussian filter. (Skip the normalisation for the filter.)
 • Convolve the image with the filter, get the res (result).
 • Create a matrix weight_origin whose elements are all ones, and apply the Gaussian Filter to thematrix, get weight. • Divide the matrix of smoothed image by the matrix of weight, get img_smoothed (smoothed image).
 • Handle the image border using partial filters in smoothing.

 (3)smooth2D()
 • call smooth1D twice and convolve each row and column with the 1D Gaussian Filter • Use img.T to transpose the image
 • 
 (4)harris() • Create a list corners to store the corner tuples.
 • Compute Ix and Iy correctly by finite differences using np.gradient. • Construct images of Ix2, Iy2, and Ix Iy correctly.
 • Smooth the squared derivatives, get Ixx, Iyy and IxxIyy. • Construct an image of the cornerness function R correctly. • Identify potential corners at local maxima in the image of the cornerness function R. • Compute the cornerness value and coordinates of the potential corners up to sub-pixel accuracy by quadratic approximation. • Use the threshold value to identify strong corners for output and append them into corners list.