# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords   

def center_face(original_image,image,shape):
    '''
    center face
    '''
    image = original_image-image   
    num_rows, num_cols = image.shape[:2]
    x_diff =  num_cols/2-(shape[0][0]+shape[16][0])/2
    y_diff = (shape[0][1]+shape[16][1])/2
   
    translation_matrix = np.float32([ [1,0, x_diff], [0,1,num_rows/2-y_diff] ])

    img_translation = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
    image = img_translation
    return image

def call_face(img,detector,predictor, centered=False, method = 'circle'):
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor


    # load the input image, resize it, and convert it to grayscale
    image = img.copy()
    image = imutils.resize(image, width=500)
    original_image = image.copy()
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
     
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
        # show the face number
        #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        if method == 'circle':
            eye1 = shape[36:42]
            eye2 = shape[42:48]
            c1 = ( int( (shape[36][0] + shape[39][0])/2 ) , int( (shape[36][1] + shape[39][1])/2 ) )  
            c2 = ( int( (shape[42][0] + shape[45][0])/2 ) , int( (shape[42][1] + shape[45][1])/2 ) )  
            cv2.circle(image,c1,5,(0, 0, 0), -1)
            cv2.circle(image,c2,20,(0, 0, 0), -1)

        elif method == 'fill':
            eye1 = shape[36:42]
            eye2 = shape[42:48]
            c1 = ( int( (shape[36][0] + shape[39][0])/2 ) , int( (shape[36][1] + shape[39][1])/2 ) )  
           
            c2 = ( int( (shape[42][0] + shape[45][0])/2 ) , int( (shape[42][1] + shape[45][1])/2 ) )  

            cv2.fillPoly(image,[eye1],(255,255,255))
            cv2.fillPoly(image,[eye2],(255,255,255))
            cv2.circle(image,c1,5,(0, 0, 0), -1)
            cv2.circle(image,c2,5,(0, 0, 0), -1)
        elif method =="display all":
            for (x, y) in shape[:26]:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        elif method == 'floating face':
            reorder = [np.concatenate([shape[:17], shape[26:22-1:-1],shape[22:17-1:-1]])]
            #reorder = [shape[26:21:-1]]
            cv2.fillPoly(image,reorder,(0,0,0))        
              
            
            
        else:
            print('Method not defined')
            cap.release() 
            cv2.destroyAllWindows() 



    if len(rects)==1 and centered:
        image = center_face(original_image,image,shape)
    return image

cap = cv2.VideoCapture(0) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



while 1: 
    ret, img = cap.read() 
    t0 = time.time()
    out = call_face(img,detector,predictor,method='display all')
    #out = img
    t1 = time.time()

    print(t1-t0)
    cv2.imshow('img',out) 
    k = cv2.waitKey(10) & 0xff
    if k == 27: 
        break
cap.release() 
cv2.destroyAllWindows() 

