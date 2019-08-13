import csv
import cv2
import os
import json
import numpy as np
from math import tan, pi

import _init_paths
from utils import visualize_prediction_boxes
import time

dir = 'cropped_img'
try:
	os.mkdir(dir)
except:
	pass

class_dict = {
	'4': 'taxi',
	'6':  'minibus',
	'2':  'motorbike',
	'10':  'emergency car',
	'11':  'emergency van',
	'8':  'rigid',
	'5': 'van',
	'9':  'truck',
	'12':  'fire engine',
	'3':  'car',
	'1':  'cyclist',
	'0':  'pedestrian',
	'7':  'bus',
}

PATH_VIDEO_FOLDER = '/home/vivacityserver6/repos/tenacity/samples/detections_program/videos'
BOXCAR_FOLDER = '/home/vivacityserver6/repos/BoxCars/'
OUTPUT_PATH = os.path.join(BOXCAR_FOLDER, 'output')
FONT = cv2.FONT_HERSHEY_SIMPLEX
PATH_VIDEO = os.path.join(PATH_VIDEO_FOLDER,'test_3.mp4')
PATH_JSON = '/home/vivacityserver6/repos/BoxCars/output/prediction_3angles_60bins_resnet_008.json'

def crop_and_save(img_name, image):
	if not os.path.exists(os.path.join(OUTPUT_PATH, 'cropped_img')):
		os.mkdir(os.path.join(OUTPUT_PATH, 'cropped_img'))
	img_name = os.path.join(OUTPUT_PATH, 'cropped_img', img_name) + '.png'
	cv2.imwrite(img_name, image)
	return image


def read_bbox(path_csv):
	box_list = []
	with open (path_csv) as csv_file:
		csv_read = csv.reader(csv_file)
		for line in csv_read:
			box_list.append(line)
	return box_list


def read_direction(path_json):
	direction_dict = {}
	with open(path_json, 'r') as json_file:
		direction_dict = json.load(json_file)
		return direction_dict

def analyse_video(cropping_img):

	vidcap = cv2.VideoCapture(PATH_VIDEO)

	img_count = 0
	frame_count = 1

	BOX_LIST = read_bbox(os.path.join(PATH_VIDEO_FOLDER, 'test_3mp4_detections_out.csv'))
	predictions = read_direction(PATH_JSON)
	while((img_count<len(BOX_LIST)) & (vidcap.isOpened() == True)):
		frame_t0 = time.time()
		success, image = vidcap.read()

		for line in BOX_LIST:
			#print(line[0], frame_count)
			#print("line ", line)
			if int(line[0]) == frame_count:
    			
				#if float(line[-1]) > 0.4:
				x = int(line[2])
				y = int(line[3])
				width = int(line[4])
				height = int(line[5])
				class_label = class_dict[line[6]]
				
				img_name = '_'.join([str(frame_count), str(img_count), class_label])
				img_cropped = image[y:y+height, x:x+width]
				crop_coordinates = np.array([x, y, width, height], dtype='int32')
				if cropping_img:
					crop_and_save(img_name, img_cropped)
				else:
					prediction_per_image = predictions[img_name]
					image = visualize_prediction_boxes(prediction_per_image=prediction_per_image, image=image, cropped_img = True, crop_coordinates = crop_coordinates)

					# directions_predictions = predictions[img_name]['output_d']
					# direction = directions_predictions.index(max(directions_predictions))
					# angle_predictions = predictions[img_name]['output_a']
					# angle = (angle_predictions.index(max(angle_predictions)) - 3) * 30
					
					# left_point = (round(x), round(y + height/2 + width/2 * tan(angle * pi / 180)))
					# right_point = (round(x+width), round(y + height/2 - width/2 * tan(angle * pi / 180)))
					# if direction == 1:
					# 	text = 'to_camera'
					# 	cv2.rectangle(image,(x, y), (x+width, y+height),(0,0,255),3)	# draw green box for 2D bbox
					# 	cv2.line(image, left_point, right_point, (0,0,255),3)	# draw green box for 2D bbox
					# 	cv2.putText(image, text ,(x, y), FONT, 1, (0,0,255), 2, cv2.LINE_AA)
					# else:
					# 	text = 'from_camera'
					# 	cv2.rectangle(image,(x, y), (x+width, y+height),(0,255,0),3)	# draw green box for 2D bbox
					# 	cv2.line(image, left_point, right_point, (0,255,0),3)	# draw green box for 2D bbox
					# 	cv2.putText(image, text ,(x, y), FONT, 1, (0,255,0), 2, cv2.LINE_AA)

				img_count += 1

			elif int(line[0]) > frame_count:
				#print('>')	#debugging purpose
				frame_count += 1
				break

		cv2.imshow('frame', image)
		show_time = int((time.time() - frame_t0)*1000)
		print((time.time() - frame_t0), show_time)
		print("FPS", 1/(time.time() - frame_t0))
		if cv2.waitKey(show_time) & 0xFF == ord('q'):
			break
		
	vidcap.release()
	cv2.destroyAllWindows()


def main():
	analyse_video(False)

if __name__ == '__main__':
    main()