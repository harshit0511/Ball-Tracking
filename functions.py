#BOWEN LIN bl2514
import cv2
import numpy as np
import math
import copy
from matplotlib import pyplot as plt


def meanintensity(img, height, width):
	totalintensity = 0
	
	for i in range (height):
		for j in range (width):
			totalintensity += img[i][j]
	return round(totalintensity / (height * width))

def deduct(img, height, width, value):
	result = np.zeros(shape = (height, width), dtype = 'int32')

	for i in range (height):
		for j in range (width):
			result[i][j] = img[i][j] - value
	return result

def matrixnorm(img):
	imgheight, imgwidth = img.shape
	sum = 0

	for i in range (imgheight):
		for j in range (imgwidth):
			sum += img[i][j] ** 2
 
	return math.sqrt(sum)

def correlation(img, filter, imgheight, imgwidth, filterheight, filterwidth):
	correlationarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'int32')
	midpointx = filterwidth // 2
	midpointy = filterheight // 2
	temp = 0

	for i in range (imgheight):
		for j in range (imgwidth):
			for y in range (-midpointy, filterheight - midpointy):
				for x in range (-midpointx, filterwidth - midpointx):
					if (i + y >=0 and i + y < imgheight) and (j + x >= 0 and j + x < imgwidth):
						temp += img[i+y][j+x] * filter[midpointy+y][midpointx+x]
			correlationarray[i][j] = temp
			temp = 0
	return correlationarray

def correlationnobound(img, filter, imgheight, imgwidth, filterheight, filterwidth):
	correlationoboundnarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'int32')
	midpointx = filterwidth // 2
	midpointy = filterheight // 2
	temp = 0

	for i in range (midpointy, imgheight - midpointy):
		for j in range (midpointx, imgwidth- midpointx):
			for y in range (-midpointy, filterheight - midpointy):
				for x in range (-midpointx, filterwidth - midpointx):
					if (i + y >=0 and i + y < imgheight) and (j + x >= 0 and j + x < imgwidth):
						temp += img[i+y][j+x] * filter[midpointy+y][midpointx+x]
			correlationoboundnarray[i][j] = temp
			temp = 0
	return correlationoboundnarray

def normalize(inputimg, img, filter, imgheight, imgwidth):
	normalizearray = np.zeros(shape = (imgheight, imgwidth), dtype = 'float64')
	temp = 0

	imgnorm = matrixnorm(img)
	filternorm = matrixnorm(filter)
	norm = imgnorm * filternorm

	for i in range (imgheight):
		for j in range (imgwidth):
			temp = inputimg[i][j] / norm
			normalizearray[i][j] = temp
	return normalizearray

def adjust(img, imgheight, imgwidth):
	adjustedarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint8')
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
	temp = 0

	for i in range(imgheight):
		for j in range (imgwidth):
			temp = (img[i][j] - min_val) / (max_val - min_val) * 255
			adjustedarray[i][j] = round(temp)
	return adjustedarray

def thresholding(img, imgheight, imgwidth, threshold):
	thresholdarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint8')

	for i in range (imgheight):
		for j in range (imgwidth):
			if img[i][j] >= threshold:
				thresholdarray[i][j] = 255
	return thresholdarray

def addtemplate(img, template, imgheight, imgwidth, templateheight, templatewidth):
	resultarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint8')
	midpointx = templatewidth // 2
	midpointy = templateheight // 2

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)

	for i in range (midpointy, imgheight - midpointy):
		for j in range (midpointx, imgwidth - midpointx):
			if img[i][j] == max_val:
				for y in range (-midpointy, templateheight - midpointy):
					for x in range (-midpointx, templatewidth - midpointx):
						resultarray[i+y][j+x] = template[midpointy+y][midpointx+x]
	return resultarray

def filter(img, filter, imgheight, imgwidth, filterheight, filterwidth):
	filterarray = copy.deepcopy(img)
	filterarray = filterarray.astype(np.float64)
	midpointx = filterwidth // 2
	midpointy = filterheight // 2
	temp = 0

	for i in range (midpointy, imgheight - midpointy):
		for j in range (midpointx, imgwidth - midpointx):
			for y in range (-midpointy, filterheight - midpointy):
				for x in range (-midpointx, filterwidth - midpointx):
					temp += img[i+y][j+x] * filter[midpointy+y][midpointx+x]
			filterarray[i][j] = temp
			temp = 0
	return filterarray

def derivative(img, filter, imgheight, imgwidth, filterheight, filterwidth):
	derivativearray = np.zeros(shape = (imgheight, imgwidth), dtype = 'int32')
	midpointx = filterwidth // 2
	midpointy = filterheight // 2
	temp = 0

	for i in range (midpointy, imgheight - midpointy):
		for j in range (midpointx, imgwidth - midpointx):
			for y in range (-midpointy, filterheight - midpointy):
				for x in range (-midpointx, filterwidth - midpointx):
					temp += img[i+y][j+x] * filter[midpointy+y][midpointx+x]
			#derivativearray[i][j] = abs(temp)
			derivativearray[i][j] = temp
			temp = 0
	return derivativearray


def magnitude(imgdx, imgdy, imgheight, imgwidth):
	magnitudearray = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint16')

	for i in range (1, imgheight - 1):
		for j in range (1, imgwidth - 1):
			magitudevalue = math.sqrt((imgdx[i][j] ** 2 + imgdy[i][j] ** 2))
			magnitudearray[i][j] = round(magitudevalue)
	return magnitudearray

def orientation(imgdx, imgdy, imgheight, imgwidth, img):
	orientationarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint8')
	edgethinarray = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint16')

	orientation = 0

	for i in range (1, imgheight - 1):
		for j in range (1, imgwidth - 1):
			orientation = math.atan2(imgdy[i][j], imgdx[i][j])
			degree = orientation / np.pi * 180
			if degree < 0:
				degree += 360
			orientationarray[i][j] = round(degree / 360 * 255)

			if (degree <= 22.5 and degree > 337.5) or (degree > 157.5 and degree <= 202.5):
				if img[i][j] < img[i][j-1] or img[i][j] < img[i][j+1]:
					edgethinarray[i][j] = 0
				else:
					edgethinarray[i][j] = img[i][j]
			elif (degree > 22.5 and degree <= 67.5) or (degree > 202.5 and degree <= 247.5):
				if img[i][j] < img[i-1][j+1] or img[i][j] < img[i+1][j-1]:
					edgethinarray[i][j] = 0
				else:
					edgethinarray[i][j] = img[i][j]
			elif (degree > 67.5 and degree <=112.5) or (degree > 247.5 and degree <= 292.5):
				if img[i][j] < img[i-1][j] or img[i][j] < img[i+1][j]:
					edgethinarray[i][j] = 0
				else:
					edgethinarray[i][j] = img[i][j]
			else:
				if img[i][j] < img[i-1][j-1] or img[i][j] < img[i+1][j+1]:
					edgethinarray[i][j] = 0
				else:
					edgethinarray[i][j] = img[i][j]

			if edgethinarray[i][j] > 255:
				edgethinarray[i][j] = 255
	return 	orientationarray, edgethinarray

def edgelink(i, j, edgehigh, edgelow):
	while (edgehigh[i][j] == 255):
		if edgehigh[i-1][j] == 0 and edgelow[i-1][j] == 255:
			edgehigh[i-1][j] = 255
			edgelink(i-1, j, edgehigh, edgelow)
		if edgehigh[i+1][j] == 0 and edgelow[i+1][j] == 255:
			edgehigh[i+1][j] = 255
			edgelink(i+1, j, edgehigh, edgelow)
		if edgehigh[i][j-1] == 0 and edgelow[i][j-1] == 255:
			edgehigh[i][j-1] = 255
			edgelink(i, j-1, edgehigh, edgelow)
		if edgehigh[i][j+1] == 0 and edgelow[i][j+1] == 255:
			edgehigh[i][j+1] = 255
			edgelink(i, j+1, edgehigh, edgelow)
		if edgehigh[i-1][j-1] == 0 and edgelow[i-1][j-1] == 255:
			edgehigh[i-1][j-1] = 255
			edgelink(i-1, j-1, edgehigh, edgelow)
		if edgehigh[i+1][j+1] == 0 and edgelow[i+1][j+1] == 255:
			edgehigh[i+1][j+1] = 255
			edgelink(i+1, j+1, edgehigh, edgelow)			
		if edgehigh[i-1][j+1] == 0 and edgelow[i-1][j+1] == 255:
			edgehigh[i-1][j+1] = 255
			edgelink(i-1, j+1, edgehigh, edgelow)
		if edgehigh[i+1][j-1] == 0 and edgelow[i+1][j-1] == 255:
			edgehigh[i+1][j-1] = 255
			edgelink(i+1, j-1, edgehigh, edgelow)
		break

def doublethreshold(img, imgheight, imgwidth, threshold_low, threshold_high):
	edgelow = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint8')
	edgehigh = np.zeros(shape = (imgheight, imgwidth), dtype = 'uint8')

	for i in range (1, imgheight - 1):
		for j in range (1, imgwidth - 1):
			if img[i][j] >= threshold_low:
				edgelow[i][j] = 255
			if img[i][j] >= threshold_high:
				edgehigh[i][j] = 255

	cv2.imwrite('threshold_low_map.jpg', edgelow)
	cv2.imwrite('threshold_high_map.jpg', edgehigh)

	for i in range (1, imgheight - 1):
		for j in range (1, imgwidth - 1):
			if edgehigh[i][j] == 255:
				edgelink(i, j, edgehigh, edgelow)

	return edgehigh