import math
import sys
import numpy as np
import cv2

def gaussianSmoothing(image):
    """
    Applies 7x7 Gaussian Filter to the image by convolution operation
    :type image: object
    """
    imageArray = np.array(image)
    gaussianArr = np.array(image)
    sum = 0

    for i in range(3, image.shape[0] - 3):
        for j in range(3, image.shape[1] - 3):
            sum = applyGaussianFilterAtPoint(imageArray, i, j)
            gaussianArr[i][j] = sum

    return gaussianArr


def applyGaussianFilterAtPoint(imageData, row, column):
    sum = 0
    for i in range(row - 3, row + 4):
        for j in range(column - 3, column + 4):
            sum += gaussian_filter[i - row + 3][j - column + 3] * imageData[i][j]

    return sum

def getGradientX(imgArr, height, width):
    """

    :param imgArr: NxM image to find the gradient
    :param height: height of the array
    :param width: width of the array
    :return: Array representing the gradient
    """
    imageData = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if liesInUnderRegion(imgArr, i, j):
                imageData[i + 1][j + 1] = None
            else:
                imageData[i + 1][j + 1] = prewittAtX(imgArr, i, j)

    return abs(imageData)


def getGradientY(imgArr, height, width):
    """
    Similar to the getGradientX function for Y
    """
    imageData = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if liesInUnderRegion(imgArr, i, j):
                imageData[i + 1][j + 1] = None
            else:
                imageData[i + 1][j + 1] = prewittAtY(imgArr, i, j)

    return abs(imageData)


def getMagnitude(Gx, Gy, height, width):
    """
    Computes the gradient magnitude by taking square root of gx-square plus gy-square
    :param Gx: xGradient of the image array
    :param Gy: yGradient of the image array
    :param height:
    :param width:
    :return: array representing edge magnitude
    """
    gradientData = np.empty(shape=(height, width))
    for row in range(height):
        for column in range(width):
            gradientData[row][column] = ((Gx[row][column] ** 2 + Gy[row][column] ** 2) ** 0.5) / 1.4142
    return gradientData


def getAngle(Gx, Gy, height, width):
    """
    Computes the edge angle by taking the tan inverse of yGradient/xGradient
    :param Gx:
    :param Gy:
    :param height:
    :param width:
    :return: integer array representing the edge angle
    """
    gradientData = np.empty(shape=(height, width))
    angle = 0
    for i in range(height):
        for j in range(width):
            if Gx[i][j] == 0:
                if Gy[i][j] > 0:
                    angle = 90
                else:
                    angle = -90
            else:
                angle = math.degrees(math.atan(Gy[i][j] / Gx[i][j]))
            if angle < 0:
                angle += 360
            gradientData[i][j] = angle
    return gradientData


def localMaximization(gradientData, gradientAngle, height, width):
    """
    Applies Non-Maxima suppression to gradient magnitude image
    :param gradientData: gradient image
    :param gradientAngle: gradient angle
    :param height:
    :param width:
    :return: the gradient magnitude image after non-maxima suppression
    """
    gradient = np.empty(shape=(height, width))
    numberOfPixels = np.zeros(shape=(256))
    edgePixels = 0

    for row in range(5, height - 5):
        for col in range(5, image[row].size - 5):
            theta = gradientAngle[row, col]
            gradientAtPixel = gradientData[row, col]
            value = 0

            # Sector - 1
            if (0 <= theta <= 22.5 or 157.5 < theta <= 202.5 or 337.5 < theta <= 360):
                if gradientAtPixel > gradientData[row, col + 1] and gradientAtPixel > gradientData[row, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 2
            elif (22.5 < theta <= 67.5 or 202.5 < theta <= 247.5):
                if gradientAtPixel > gradientData[row + 1, col - 1] and gradientAtPixel > gradientData[
                    row - 1, col + 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 3
            elif (67.5 < theta <= 112.5 or 247.5 < theta <= 292.5):
                if gradientAtPixel > gradientData[row + 1, col] and gradientAtPixel > gradientData[row - 1, col]:
                    value = gradientAtPixel
                else:
                    value = 0

            # Sector - 4
            elif 112.5 < theta <= 157.5 or 292.5 < theta <= 337.5:
                if gradientAtPixel > gradientData[row + 1, col + 1] \
                        and gradientAtPixel > gradientData[row - 1, col - 1]:
                    value = gradientAtPixel
                else:
                    value = 0

            gradient[row, col] = value

            # If value is greater than one after non maxima suppression
            if value > 0:
                edgePixels += 1
                try:
                    numberOfPixels[int(value)] += 1
                except:
                    print('Out of range gray level value', value)

    print('Number of Edge pixels:', edgePixels)
    return [gradient, numberOfPixels, edgePixels]


def pTile(percent, imageData, numberOfPixels, edgePixels, file):
    """
    Applies p-tile method of automatic thresholding to find the best threshold value and then apply that
    to the image to create a binary image.
    :param percent: of non zero pixels to be over the threshold
    :param imageData: input image array
    :param numberOfPixels: counts total number of pixels in the image
    :param edgePixels: counts pixels present at the edges
    :param file:
    :return: binary image array with p-tile method thresholding applied
    """
    # Number of pixels to keep
    threshold = np.around(edgePixels * percent / 100)
    sum, value = 0, 255
    for value in range(255, 0, -1):
        sum += numberOfPixels[value]
        if sum >= threshold:
            break

    for i in range(imageData.shape[0]):
        for j in range(imageData[i].size):
            if imageData[i, j] < value:
                imageData[i, j] = 0
            else:
                imageData[i, j] = 255

    print('For', percent, '- result:')
    print('Total pixels after thresholding:', sum)
    print('Threshold gray level value:', value)
    #     plt.imshow(imageData, cmap='gray')
    cv2.imwrite('Outputs/' + str(percent) + "_percent.jpg", imageData)

def liesInUnderRegion(imgArr, i, j):
    return imgArr[i][j] == None or imgArr[i][j + 1] == None or imgArr[i][j - 1] == None or imgArr[i + 1][j] == None or \
           imgArr[i + 1][j + 1] == None or imgArr[i + 1][j - 1] == None or imgArr[i - 1][j] == None or \
           imgArr[i - 1][j + 1] == None or imgArr[i - 1][j - 1] == None

def prewittAtX(imageData, row, column):
    sum = 0
    horizontal = 0
    for i in range(0, 3):
        for j in range(0, 3):
            horizontal += imageData[row + i, column + j] * prewittX[i, j]
    return horizontal

def prewittAtY(imageData, row, column):
    sum = 0
    vertical = 0
    for i in range(0, 3):
        for j in range(0, 3):
            vertical += imageData[row + i, column + j] * prewittY[i, j]
    return vertical

if __name__ == "__main__":

    gaussian_filter = (1.0 / 140.0) * np.array([[1, 1, 2, 2, 2, 1, 1],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [2, 4, 8, 16, 8, 4, 2],
                                                [2, 2, 4, 8, 4, 2, 2],
                                                [1, 2, 2, 4, 2, 2, 1],
                                                [1, 1, 2, 2, 2, 1, 1]])

    prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]])

    prewittY = (1.0 / 3.0) * np.array([[1, 1, 1],
                                       [0, 0, 0],
                                       [-1, -1, -1]])

    # file = sys.argv[1]
    # Read Image and convert to 2D numpy array
    image = cv2.imread('Images/Lena256.bmp', 0)

    height = image.shape[0]
    width = image.shape[1]

    # Normalized Gaussian Smoothing
    gaussianData = gaussianSmoothing(image)
    print(gaussianData)
    cv2.imwrite('Outputs/filter_gauss.jpg', gaussianData)

    # Normalized Horizontal Gradient
    Gx = getGradientX(gaussianData, height, width)
    cv2.imwrite('Outputs/XGradient.jpg', Gx)

    # Normalized Vertical Gradient
    Gy = getGradientY(gaussianData, height, width)
    cv2.imwrite('Outputs/YGradient.jpg', Gy)

    # Normalized Edge Magnitude
    gradient = getMagnitude(Gx, Gy, height, width)
    cv2.imwrite('Outputs/Gradient.jpg', gradient)

    # Edge angle
    gradientAngle = getAngle(Gx, Gy, height, width)
    # print(gradientAngle.shape)

    # Non maxima suppression
    localMaxSuppressed = localMaximization(gradient, gradientAngle, height, width)
    cv2.imwrite('Outputs/MaximizedImage.jpg', localMaxSuppressed[0])

    suppressedImage = localMaxSuppressed[0]
    numberOfPixels = localMaxSuppressed[1]
    edgePixels = localMaxSuppressed[2]

    # Binary Edge image with p-tile Threshold 10%
    ptile10 = pTile(10, np.copy(suppressedImage), numberOfPixels, edgePixels, image)

    # Binary Edge image with p-tile Threshold 20%
    ptile20 = pTile(30, np.copy(suppressedImage), numberOfPixels, edgePixels, image)

    # Binary Edge image with p-tile Threshold 30%
    ptile30 = pTile(50, np.copy(suppressedImage), numberOfPixels, edgePixels, image)
