	#!/usr/bin/python
	# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import normalize
import cv2
import imageio
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from StereoLab_Fine_Tuning import Depth_Estimation
from mlxtend.image import extract_face_landmarks
from scipy.spatial import distance
import sys

	# Creating 3D Landmarks using 2D data and Depth values which are estimated using the initial steps

def Landmarks3D(landmarks, z_ax):
	land3d = []
	xdata = []
	ydata = []
	zdata = []
	for r in landmarks:
	    if r[0] < len(z_ax) and r[1] < len(z_ax):
	        xdata.append(r[0])
	        ydata.append(r[1])
	        #print(r[0],r[1])
	        zdata.append(z_ax[r[0]][r[1]] * 40)
	        land3d.append([r[0], r[1], z_ax[r[0]][r[1]]])
	return (land3d, xdata, ydata, zdata)


# Plot the landmarks for better idea

def Plot3D(xdata, ydata, zdata):
	ax = plt.axes(projection='3d')
	ax.scatter3D(xdata, ydata, zdata, c=zdata)

	# ax.plot(ydata,xdata,zdata, color='r')

	plt.show()


def main():

	print('loading stock face data...')
	stock_data = sys.argv[1]
	print(stock_data)

	
	imgL = cv2.imread(stock_data+'L.jpg')  # downscale images for faster processing
	imgR = cv2.imread(stock_data+'R.jpg')
	imgL = cv2.resize(imgL, (400, 400), interpolation=cv2.INTER_AREA)
	imgR = cv2.resize(imgR, (400, 400), interpolation=cv2.INTER_AREA)
	landmarks = extract_face_landmarks(imgL)
	#print(len(landmarks))
	z_ax = Depth_Estimation(
	    imgL,
	    imgR,
	    5,
	    7,
	    55,
	    True,
	    )
	#print(len(z_ax))
	(land3d1, xdata1, ydata1, zdata1) = Landmarks3D(landmarks, z_ax)
	#Plot3D(xdata1,ydata1,zdata1)
	print('loading input face data...')
	input_data = sys.argv[2]
	print(input_data)

	imgL = cv2.imread(input_data+'L.jpg')  # downscale images for faster processing
	imgR = cv2.imread(input_data+'R.jpg')

	imgL = cv2.resize(imgL, (400, 400), interpolation=cv2.INTER_AREA)
	imgR = cv2.resize(imgR, (400, 400), interpolation=cv2.INTER_AREA)
	landmarks = extract_face_landmarks(imgL)
	#print(landmarks)
	z_ax = Depth_Estimation(
	    imgL,
	    imgR,
	    5,
	    7,
	    55,
	    True,
	    )
	#print(z_ax)
	(land3d2, xdata2, ydata2, zdata2) = Landmarks3D(landmarks, z_ax)
	#Plot3D(xdata2,ydata2,zdata2)
	res = 0
	ln = min(len(land3d1), len(land3d2))

	for i in range(0, ln):
		if land3d2[i][2] != float("inf") and land3d1[i][2] != float("inf"):
			d1 = np.asarray([land3d1[i][0],land3d1[i][1],land3d1[i][2]])
			d2 = np.asarray([land3d2[i][0],land3d2[i][1],land3d2[i][2]])
			res += distance.euclidean(d1, d2)

	# Best possible accuracy achieved by us is 2000, This is becuase of the collected data is not Stereo Accurate.

	# But, Sri data gave a great accuracy of less than 1000 residue.

	# With Roh data we did achieve at accuracy of 900 as well. But, with only certain stock and input combinations. Which is not reliable in real world sometimes.

	if res < 2000:
	    print('Face Recognised! and Residue is:', res)
	else:
	    print('No Match. Becuase residue is:', res)


if __name__ == '__main__':
	main()
