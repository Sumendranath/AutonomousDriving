import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#matplotlib inline

#reading in an image
dir_img = 'test_images/'
image = mpimg.imread('test_images/medium.png')#RGB
#print out some stats and plotting
print('This image is:',type(image),'with dimensions:',image.shape)
plt.imshow(image)
plt.show()
