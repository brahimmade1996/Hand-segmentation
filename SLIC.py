# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
# import argparse
import cv2

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "hand.jpg")
# args = vars(ap.parse_args())
# image = cv2.imread('blond_face.jpeg')

# load the image and convert it to a floating point data type
image = img_as_float(io.imread('cropped_blond.jpg'))
# r = cv2.selectROI('img', image)
# imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# cv2.imwrite('cropped_blond.jpg', imCrop)

# loop over the number of segments
for numSegments in (1000, 2000):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(image, n_segments = numSegments, sigma = 5)

	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	markedIm = mark_boundaries(image, segments, (0,0,1))

	if numSegments == 1000:
		io.imsave("segmented_face_1.jpg", markedIm)
	ax.imshow(markedIm)
	plt.axis("off")

# show the plots
plt.show()

# apply SLIC and extract (approximately) the supplied number
# 	# of segments
# segments = slic(image, n_segments = numSegments, sigma = 5)