from imutils import face_utils
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import dlib


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img = cv2.imread('images/cropped_face.jpg')
image = imutils.resize(img, width=500)
# mask = np.zeros(image.shape[:2],np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
  # determine the facial landmarks for the face region, then
  # convert the facial landmark (x, y)-coordinates to a NumPy
  # array
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)

  # # convert dlib's rectangle to a OpenCV-style bounding box
  # # [i.e., (x, y, w, h)], then draw the face bounding box
  # (x, y, w, h) = face_utils.rect_to_bb(rect)
  # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # # show the face number
  # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
  #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  # loop over the (x, y)-coordinates for the facial landmarks
  # and draw them on the image
  for (x, y) in shape[29:36]:
    cv2.circle(gray, (x, y), 1, (0, 0, 0), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", gray)
cv2.waitKey(0)



mask = np.zeros(image.shape[:2],np.uint8)

mask[gray == 0] = 0
mask[gray == 255] = 1

# # plt.imshow(img)
# # plt.show()

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
 
# rect = (94,250,697,967)
cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image = image*mask2[:,:,np.newaxis]
 
plt.imshow(image),plt.colorbar(),plt.show()