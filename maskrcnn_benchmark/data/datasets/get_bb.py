import cv2
import numpy as np
import xml.etree.ElementTree
# import math

'''
returns contours and a mask of seeds for an given image
''' 
def get_contours(image_name): 	
	# import os
	image = cv2.imread(image_name)
	cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	ret, maskR = cv2.threshold(image[:,:,0], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	ret, maskG = cv2.threshold(image[:,:,1], 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
	ret, maskB = cv2.threshold(image[:,:,2], 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
	mask = cv2.bitwise_and(maskR,maskB)
	# Find Contours
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours, mask


'''
returns coordinates of predicted (computer vision only) bounding boxes, does not provide a label 
'''
def get_bb(image_name):
	image = cv2.imread(image_name)
	cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	contours, mask = get_contours(image_name)
	good_cnt = []
	for cnt in contours:          
		if cv2.contourArea(cnt) < 1000: continue
		if cv2.contourArea(cnt) > 100000: continue
		if cv2.arcLength(cnt,False) > 2000.0: continue
		if (cv2.arcLength(cnt,False)/cv2.contourArea(cnt)) > 0.2: continue
#         print(1)
		good_cnt.append(cnt)
		
	b_boxes = []    
	for cnt in good_cnt:
		x,y,w,h = cv2.boundingRect(cnt)
		b_boxes.append((x,y, x + w, y + h))
#         print(1)
	return b_boxes, good_cnt, mask

# Returns a list of bounding boxes in tuples: (xmin, ymin, xmax, ymax) + labels for a tuple
def get_ground_bb(xml_name):  
	# image = cv2.imread(image_name)
	# # cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	print(xml_name)
	root = xml.etree.ElementTree.parse(xml_name).getroot()
#     gernimated_string = 'germinated'
#     non_gernimated_string = 'non-germinated'
	objects = root.findall('object')
	b_boxes_xml = [obj.find('bndbox') for obj in objects]
	labels = [obj.find('name').text for obj in objects]
	b_boxes = [(int(box.find('xmin').text), int(box.find('ymin').text), int(box.find('xmax').text), int(box.find('ymax').text)) for box in b_boxes_xml] 
	return b_boxes, labels

def get_ground_bb_2(xml_name_seed, xml_name_rad , override = True):  
	print(xml_name_rad)
	root_seed = xml.etree.ElementTree.parse(xml_name_seed).getroot()
	objects_seed = root_seed.findall('object')

	root_rad = xml.etree.ElementTree.parse(xml_name_rad).getroot()
	objects_rad = root_rad.findall('object')

	objects = objects_seed +  objects_rad 

	labels = [obj.find('name').text for obj in objects]
			# labels_xml = labels_xml + labels_xml_rad 
	b_boxes_xml = [obj.find('bndbox') for obj in objects]
	boxes = [(int(box.find('xmin').text), int(box.find('ymin').text), int(box.find('xmax').text), int(box.find('ymax').text)) for box in b_boxes_xml]
	if override:
		labels = ['seed'] * len(objects_seed) + ['radical'] * len(objects_rad)
		# print(labels) 
	return boxes, labels

def save_image(image_name, xml_name, b_boxes):
	image = cv2.imread(image_name)
	cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
	ground_boxes = get_ground_bb(image_name, xml_name)
	for box in b_boxes:
		(xmin, ymin, xmax, ymax) = box
		cv2.rectangle(image, (xmin,ymin), (xmax, ymax), (255,0,0), 4)
	for box in ground_boxes:
		(xmin, ymin, xmax, ymax) = box
		cv2.rectangle(image, (xmin,ymin), (xmax, ymax), (0, 0,255), 4)
	cv2.imwrite(f'image{index}.png', image)