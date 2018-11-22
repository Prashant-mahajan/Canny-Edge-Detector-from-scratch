# Canny-Edge-Detector-from-scratch

![Input Image](https://github.com/Prashant-mahajan/Canny-Edge-Detector-from-scratch/blob/master/Images/Lena256.png) | ![Output Image](https://github.com/Prashant-mahajan/Canny-Edge-Detector-from-scratch/blob/master/Outputs/Lena/50_percent.jpg)
	
This repository contains implementation of Canny Edge Detector from scratch without using library functions (except for image open/close & matrix operations) in Python. 

Canny Edge Detection is an edge detection operator that is uses a multistage algorithm to detect a wide range of edges in images. 
For more information click [here](https://en.wikipedia.org/wiki/Canny_edge_detector). 

The main steps in this algorithms are as follows: 
1. Grayscale Conversion
2. Gaussian Blur
3. Determing the Intensity Gradients
4. Non Maxima Suppression
5. Double Thresholding 
6. Edge Tracking by Hysteresis 

For detailed explanation of each step click [here](http://justin-liang.com/tutorials/canny/). 











