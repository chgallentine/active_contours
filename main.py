# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @Date:   2020-06-28 19:52:40
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2020-06-28 19:56:54

import cv2
import numpy as np 
from time import sleep

from active_contour import *

# Row/Column format
contour_points = [(88,156),(144,151),(170,110),(205,155),(241,155),(189,224),(135,225)]

img = np.array(cv2.imread('star.jpg',0))
lap = np.array(img_laplacian(img))

contour = Contour(contour_points)

# Add a ton more points
contour.insert_points()
contour.insert_points()
contour.insert_points()
contour.insert_points()
contour.insert_points()

# Create series of images fitting contour
allimgs = []
for i in range(100):
	lapcpy = np.copy(lap)
	contour.calc_energies(lapcpy)

	contour.update_points()

	contour.draw_contour(lapcpy)

	allimgs.append(lapcpy)

	# cv2.imwrite('./imgs/%d.png' % i,lapcpy)

# Creates a short animation of curve fitting
breakon = False
while True:
	for img in allimgs:
		img = cv2.resize(img, (550,550))
		cv2.imshow('Image', img)

		sleep(0.01)

		k = cv2.waitKey(33)
		if k==27:    # Esc key to stop
			breakon = True

		if breakon:
			break

	if breakon:
			break














