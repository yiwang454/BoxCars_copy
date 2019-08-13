import cv2
import numpy as np

def image_preprocess(im):
    
    desired_size = 224

    old_size = im.shape[:2]  # im.shape is in (height, width, channel) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    # create a new image and paste the resized on it
    new_im = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
    offset_x = (desired_size-new_size[1])//2
    offset_y = (desired_size-new_size[0])//2
    new_im[offset_y: offset_y + new_size[0], offset_x: offset_x + new_size[1]] = im
    
    return new_im

im = cv2.imread('/home/vivacityserver6/repos/BoxCars/output/cropped_img/1_0_car.png')
cv2.imwrite('./test_origin.jpg', im)
image = image_preprocess(im)
cv2.imwrite('./test.jpg', image)