from imutils import face_utils
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import dlib

def drawPolyline(im, landmarks, start, end, isClosed=False):
  points = []
  for i in range(start, end+1):
    point = [landmarks.part(i).x, landmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(im, [points], isClosed, (255, 200, 0), thickness=2)

# Use this function for 70-points facial landmark detector model
def renderFace(im, landmarks):
    assert(landmarks.num_parts == 68)
    # drawPolyline(im, landmarks, 0, 16)           # Jaw line
    # drawPolyline(im, landmarks, 17, 21)          # Left eyebrow
    # drawPolyline(im, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im, landmarks, 27, 30)          # Nose bridge
    # drawPolyline(im, landmarks, 30, 35, True)    # Lower nose
    # drawPolyline(im, landmarks, 36, 41, True)    # Left eye
    # drawPolyline(im, landmarks, 42, 47, True)    # Right Eye
    # drawPolyline(im, landmarks, 48, 59, True)    # Outer lip
    # drawPolyline(im, landmarks, 60, 67, True)    # Inner lip



if __name__ == '__main__':
  # initialize dlib's face detector (HOG-based) and then create
  # the facial landmark predictor
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

  img = cv2.imread('images/cropped_blond.jpg')
  image = imutils.resize(img, width=500)
  # mask = np.zeros(image.shape[:2],np.uint8)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(hsv)

  # detect faces in the grayscale image
  rects = detector(gray, 1)
  intensity = 0

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
    for (x, y) in shape[27:30]:
      intensity = V[x,y]
    #   cv2.circle(gray, (x, y), 1, (0, 0, 0), -1)
    # renderFace(gray, shape)

  # show the output image with the face detections + facial landmarks
  print(intensity)
  # cv2.imshow("Output", V)
  # cv2.waitKey(0)



  mask = np.zeros(image.shape[:2],np.uint8)


  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      diff = abs(V[i,j] - intensity)
      if diff < 20:
        mask[i,j] = 1
      elif diff > 20 and diff < 30:
        mask[i,j] = 3
      elif diff > 30 and diff < 40:
        mask[i,j] = 2
      else:
        mask[i,j] = 0

  # # plt.imshow(img)
  # # plt.show()

  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
   
  # rect = (94,250,697,967)
  cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  grabcut_image = image*mask2[:,:,np.newaxis]

  combined = np.hstack((image, grabcut_image))
  # cv2.imwrite('Grabcut Result', combined)
  # cv2.imshow('Grabcut Result', combined)
  # cv2.waitKey(0)
  plt.imshow(grabcut_image),plt.colorbar(),plt.show()