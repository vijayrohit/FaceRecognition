#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import normalize
import imageio
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
from mlxtend.image import extract_face_landmarks
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import glob


# Use of the StereoSGBM_create, createDisparityWLSFilter, setLambda, setSigmaColor, and compute is adapted from https://timosam.com/python_opencv_depthimage/
# But, fine tuning is done by us. Fine tuned depth map is an essential requirement for face recognition.

def Depth_Estimation(
    imgL,
    imgR,
    window_size,
    f,
    b,
    flag):
    left_matcher = cv2.StereoSGBM_create(
        numDisparities=16 * f,
        blockSize=7,
        P1=8 * 4 * window_size ** 2,
        P2=32 * 4 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=16,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    # -----------Adapting similar filter structure to Right Matcher as well-------------#
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters

    lmbda = 80000
    sigma = 0.8
    visual_multiplier = 1


    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    d = np.subtract(dispr, displ)

    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg,
                                beta=b, alpha=105,
                                norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    if flag == True:
    	z_ax = Z_Values(d)
    	return z_ax
    else:
    	return filteredImg



#Z (Depth) Coordinates
def Z_Values(d):
	z_ax = []
	mx = 0
	for r in d:
		row = []
		for c in r:
			if c!=0:
				#Focal length of our lens is 25mm and 100mm of base
				z = float(2500/c)
				if z > mx:
					mx =z
			else:
				z = float('inf')
			row.append(z)
		z_ax.append(row)
	return z_ax




def main():
    subdir = os.listdir('SELF_COLLECTED_DATA/')

    for dirc in subdir:
        print('loading images...')

        imgL = cv2.imread('SELF_COLLECTED_DATA/' + dirc + '/L.jpg')
        imgR = cv2.imread('SELF_COLLECTED_DATA/' + dirc + '/R.jpg')
        imgL = cv2.resize(imgL, (400, 400), interpolation=cv2.INTER_AREA)
        imgR = cv2.resize(imgR, (400, 400), interpolation=cv2.INTER_AREA)
        # ------Parameters Initialized---------------#

        window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        f = 3
        b = 15
        while True:

            filteredImg = Depth_Estimation(imgL, imgR, window_size, f,
                    b,False)

            # ------------Checked all possible conditions around 40 and cut down to these combinations to show how tuning them gave a better depth map--------#

            if b < 55:
                b = b + 5
            if b == 25 and f == 3:
                f = f + 1
            if b == 35 and f == 4:
                f = f + 1
            if b == 45 and f == 5:
                f = f + 1
            if b == 55:
                window_size = window_size + 2
                f = f + 1
            if window_size == 5:
                filteredImg = Depth_Estimation(imgL, imgR, window_size,
                        f, b,False)
                cv2.imwrite('SELF_COLLECTED_DATA/' + dirc + '/test_'
                            + str(window_size) + '_' + str(f) + '_'
                            + str(b) + '.png', filteredImg)
                break
            cv2.imwrite('SELF_COLLECTED_DATA/' + dirc + '/test_'
                        + str(window_size) + '_' + str(f) + '_'
                        + str(b) + '.png', filteredImg)

    
if __name__ == '__main__':
    main()
