from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def con(scratch,i,j):
    gauss_filter = np.array([[1, 1, 2, 2, 2, 1, 1],
              [1, 2, 2, 4, 2, 2, 1],
              [2, 2, 4, 8, 4, 2, 2],
              [2, 4, 8, 16, 8, 4, 2],
              [2, 2, 4, 8, 4, 2, 2],
              [1, 2, 2, 4, 2, 2, 1],
              [1, 1, 2, 2, 2, 1, 1]])
    sum=0
    for k in range(i-3,i+ 4):
        for l in range(j-3, j+4):
            #sc = scratch[i + l, j + k]      # used to obatin the respective image pixels for performing convolution
            #fil = gauss_filter[l+3, k+3 ]   # used to obatin the corresponding gaussian mask pixel
            sum+=gauss_filter[k-i+3][l-j+3]*scratch[k][l]
    return (sum/140)                       # returns the normalized gaussian filtered value for the pixel [i,j]


def prewitt(im2,i,j):
    h,v=0,0
    result=[]
    Gx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    Gy = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
    for k in range(-1, 1):
        for l in range(-1, 1):
            sc = im2[i + k][j + l]
            a, b = Gx[1 + k][1 + l], Gy[1 + k][1 + l]

            h+= abs(sc * a)
            v+= abs(sc * b)
            #print(h)
            #print(v)
    result.append(h)
    result.append(v)
    return result


def grad(im_hori,im_verti,i,j):          # Function for calculating the Gradient Magnitude and direction
    result1=[]
    result1.append(np.power(np.power(im_verti[i][j], 2.0) + np.power(im_hori[i][j], 2.0), 0.5))
    result1.append(np.arctan2(im_verti[i][j], im_hori[i][j]))
    result1.append((np.round(result1[1] * (5.0 / np.pi)) + 5) % 5)          # Quantize direction
    return result1


def Nonmax_sup(im_mag,tq,r,c):
    # Suppress pixels at the image edge
    if tq == 0:  # 0 is E-W (horizontal)
        if im_mag[r, c] <= im_mag[r, c - 1] or im_mag[r, c] <= im_mag[r, c + 1]:
            return 0
        else: return 1
    elif tq == 1:  # 1 is NE-SW
        if im_mag[r, c]!=max(im_mag[r - 1, c + 1],im_mag[r, c],im_mag[r + 1, c - 1]):
            return 0
        else: return 1
    elif tq == 2:  # 2 is N-S (vertical)
        if im_mag[r, c]!=max(im_mag[r - 1, c],im_mag[r, c],im_mag[r + 1, c]):
            return 0
        else: return 1
    elif tq == 3:  # 3 is NW-SE
        if im_mag[r, c]!=max(im_mag[r - 1, c - 1],im_mag[r, c],im_mag[r + 1, c + 1]):
            return 0
        else: return 1

def p_tile(over,a,p):
    #p=100-p
    count=0
    a = np.array(a)                      # return pth percentile
    t=np.percentile(over,p)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if(a[i][j]>=t):
                count+=1
                a[i][j]=255                    # Assigning value 255 to the foreground pixels
            else : a[i][j]=0                   # Assigning value 0 to the background pixels
    print("Treshold: ",t)
    return a


def normalized(arr, value):

    absoluteArr = np.array(arr)

    # Take absolute value
    for i in range(absoluteArr.shape[0]):
        for j in range(absoluteArr.shape[1]):
            absoluteArr[i][j] = abs(arr[i][j])

    maxVal = -sys.maxsize
    minVal = sys.maxsize

    for i in range(absoluteArr.shape[0]):
        for j in range(absoluteArr.shape[1]):
            maxVal = max(absoluteArr[i][j], maxVal)
            minVal = min(absoluteArr[i][j], minVal)

    result = np.array(arr)
    normalizeImage(absoluteArr, value, maxVal, minVal, result)

    return result


def normalizeImage(arr, high, maxVal, minVal, result):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != None:
                result[i][j] = int(((arr[i][j] - minVal) / float(maxVal - minVal)) * high)
            else:
                result[i][j] = 0

class Project:
    img = Image.open("zebra-crossing-1.bmp")
    # img.show()
    w,h=img.size
    temp = np.array(img, float)
    #plt.imshow(temp)

    # Starting the Edge Detector with Gaussian Filtering
    im2 = np.array(img)
    scratch = np.pad(img, [3, 3], mode='constant') # to handle the pixels for which gaussian mask goes outside image border
    print("Original image")
    print(img)
    #print("padded image")
    #print(scratch)

    for i in range(3,h-3):
        for j in range(3,w-3):
            r=con(temp,i,j)  # returns the dicsrete convolution for pixel[i,j]
            im2.itemset((i, j), r) # stores the result of gaussian smoothing into
    figure, axis = plt.subplots(1, 2)
    axis[0].set_title('Originl Image')
    axis[0].imshow(img,cmap='gray')
    axis[1].set_title('Smoothened Image')
    axis[1].imshow(im2,cmap='gray')
    plt.show()

    # Moving on to Prewitt's Operator
    im_hori,im_verti=np.array(im2),np.array(im2)
    #im_hori = np.pad(im_hori, [3, 3], mode='constant')
    #im_verti = np.pad(im_verti, [3, 3], mode='constant')

    result=[]

    for i in range(h):
        for j in range(w):
            result=prewitt(im2,i,j)
            im_hori.itemset((i,j),result[0])  # Storing Horizontal Gradient
            im_verti.itemset((i,j),result[1]) # Storing Vertical Gradient
    im_hori1=normalized(im_hori, 255)
    im_verti1=normalized(im_verti,255)


    figure2, axis = plt.subplots(1, 2)
    axis[0].set_title("Horizontal Gradient")
    axis[0].imshow(im_hori1, cmap='gray')
    axis[1].set_title("Vertical Gradient")
    axis[1].imshow(im_verti1, cmap='gray')
    plt.show()
    print("hori magnitude")
    print(im_hori)
    print("verti magnitude")
    print(im_verti)

    # Moving on to find the Magnitude and Direction
    # Get gradient and direction

    im_mag,im_angle,im_sector=np.array(im_hori),np.array(im_verti),np.array(im_verti) # Initializing the Magnitude, Angle and Sector image arrays
    h,w=im_hori.shape[0],im_hori.shape[1]
    result1=[]
    for i in range(h):
        for j in range(w):
            result1=grad(im_hori,im_verti,i,j) # Calculating Magnitude and Gradient Angle
            im_mag.itemset((i,j),result1[0])  # Storing Horizontal Gradient
            im_angle.itemset((i,j),result1[1]) # Storing Vertical Gradient
            im_sector.itemset((i,j),result1[2]) # Storing the sector for each pixel
    im_mag1=normalized(im_mag,255)
    plt.title("Gradient Magnitude")
    plt.imshow(im_mag1, cmap='gray')
    plt.show()
    print("Gradient magnitude")
    print(im_mag)
    print(im_angle)

    # Non-maximum suppression
    gradSup = np.array(im_mag1)
    print("printing gradSup for the first time")
    print(gradSup)
    for r in range(temp.shape[0]):
        for c in range(temp.shape[1]):
            # Suppress pixels at the image edge
            if r == 0 or r == temp.shape[0] - 1 or c == 0 or c == temp.shape[1] - 1:
                gradSup[r, c] = 0
                continue
            tq=im_sector[r, c]% 4
            if(Nonmax_sup(im_mag1,tq,r,c)==0):
                gradSup[r][c]=0
    print("printing gradSup for the second time")
    print(gradSup)
    gradSup=normalized(gradSup,255)
    plt.title("Non-Maxima Supression")
    plt.imshow(gradSup, cmap="gray")
    plt.show()


    # Starting with P-tile Thresholding
    gradSup1,gradSup2,gradSup3=gradSup.copy(),gradSup.copy(),gradSup.copy()
    over = []
    for i in range(gradSup.shape[0]):
        for j in range(gradSup.shape[1]):
            if gradSup[i][j] > 0:
                over.append(gradSup[i][j])
    over.sort()
    print(over)
    print("gradSup")
    print(gradSup)
    print("gradSup1")
    print(gradSup1)
    print("gradSup2")
    print(gradSup2)
    print("gradSup3")
    print(gradSup3)
    Pt1 = p_tile(over,gradSup1,10)            # Tresholding
    plt.title("P-tile Thresholding (P=10)")
    plt.imshow(Pt1, cmap="gray")
    plt.show()
    Pt2 = p_tile(over,gradSup2,30)
    plt.title("P-tile Thresholding (P=30)")
    plt.imshow(Pt2, cmap="gray")
    plt.show()
    Pt3 = p_tile(over,gradSup3,50)
    plt.title("P-tile Thresholding (P=50)")
    plt.imshow(Pt3, cmap="gray")
    plt.show()


