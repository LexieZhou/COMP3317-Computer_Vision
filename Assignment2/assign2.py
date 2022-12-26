################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    #r, g, b = img_color[:,:,0], img_color[:,:,1], img_color[:,:,2]
    #img_gray = 0.299*r + 0.587*g + 0.144*b
    height, width = img_color.shape[0], img_color.shape[1]
    img_gray = np.zeros((height, width))

    for y in range(0,height):
        for x in range(0, width):
            r, g, b = img_color[y, x]
            img_gray[y][x] = 0.299*r + 0.587*g + 0.114*b

    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    # sigma = 1 the kernel need to be truncated
    n = int(sigma * (2 * np.log(1000)) ** 0.5)
    x = np.arange(-n, n+1)
    gaussian_filter = np.exp ((x ** 2) / -2 / (sigma ** 2))
    gaussian_filter1 = np.extract(gaussian_filter >= 1/1000*np.exp(0), gaussian_filter)

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    res = convolve1d(img, gaussian_filter1, 1, np.float64, 'constant', 0, 0)

    height = res.shape[0]
    width = res.shape[1]
    weight_origin = np.ones((height, width))
    weight = convolve1d(weight_origin, gaussian_filter1, 1, np.float64, 'constant', 0, 0)

    img_smoothed = np.divide(res, weight)

    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed0 = smooth1D(img, sigma)

    # TODO: smooth the image along the horizontal direction
    img_smoothed1 = smooth1D(img_smoothed0.T, sigma)
    img_smoothed = img_smoothed1.T

    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner
    corners = []
    # TODO: compute Ix & Iy
    Ix = np.gradient(img, axis = 0)
    Iy = np.gradient(img, axis = 1)

    # TODO: compute Ix2, Iy2 and IxIy
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = np.multiply(Ix,Iy)

    # TODO: smooth the squared derivatives
    Ixx = smooth2D(Ix2, sigma)
    Iyy = smooth2D(Iy2, sigma)
    IxxIyy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness functoin R
    height, width = img.shape[0], img.shape[1]
    R = np.zeros((height, width))

    k=0.04
    detA = Ixx * Iyy - IxxIyy**2
    traceA = Ixx + Iyy
    R = detA - k * traceA **2

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy
    for x in range(1, height-1):
        for y in range(1, width-1):
            arr = np.array([R[x-1, y-1], R[x-1, y], R[x-1, y+1], R[x, y-1], R[x, y], R[x, y+1], R[x+1, y-1], R[x+1, y], R[x+1, y+1]])
            if R[x, y] == np.amax(arr):
                a = (R[x-1, y] + R[x+1, y] - 2*R[x, y])/2
                b = (R[x, y-1] + R[x, y+1] - 2*R[x, y])/2
                c = (R[x+1, y] - R[x-1, y])/2
                d = (R[x, y+1] - R[x, y-1])/2

    # TODO: perform thresholding and discard weak corners
                if (R[x, y] > threshold):
                    if (a!= 0) and (b!=0):
                        xe = x-c/(2*a)
                        ye = y-d/(2*b)
                        corners.append([ye, xe, R[x, y]])

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#  show corner detection result
################################################################################
def show_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]

    plt.ion()
    fig = plt.figure('Harris corner detection')
    plt.imshow(img_color)
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  save corners to a file
################################################################################
def save_corners(outputfile, corners) :
    # input:
    #    outputfile - path of the output file
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(outputfile, 'w')
        file.write('{}\n'.format(len(corners)))
        for corner in corners :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(corner[0], corner[1], corner[2]))
        file.close()
    except :
        print('Error occurs in writing output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load corners from a file
################################################################################
def load_corners(inputfile) :
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading {} corners'.format(nc))
        corners = np.zeros([nc, 3], dtype = np.float64)
        for i in range(nc) :
            line = file.readline()
            x, y, r = line.split()
            corners[i] = [np.float64(x), np.float64(y), np.float64(r)]
        file.close()
        return corners
    except :
        print('Error occurs in loading corners from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0,
                        help = 'sigma value for Gaussain filter (default = 1.0)')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6,
                        help = 'threshold value for corner detection (default = 1e6)')
    parser.add_argument('-o', '--output', type = str, 
                        help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : {}'.format(args.image))
    print('sigma      : {:.2f}'.format(args.sigma))
    print('threshold  : {:.2e}'.format(args.threshold))
    print('output file: {}'.format(args.output))
    print('------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('{} corners detected...'.format(len(corners)))
    show_corners(img_color, corners)

    # save corners to a file
    if args.output :
        save_corners(args.output, corners)
        print('corners saved to \'{}\'...'.format(args.output))

if __name__ == '__main__' :
    main()
