import cv2
import numpy as np
import os
import facerecognitation as fr

test_img = cv2.imread('C:/Users/a/Desktop/face rec/test_image/capture1.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected : ",faces_detected)

#for (x,y,w,h) in faces_detected:
 #   cv2.rectangle(test_img,(x,y),(w+x,h+y),(255,0,0),thickness=5)

#resized_image=cv2.resize(test_img,(1000,700))
#cv2.imshow("face detection tutorial",resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

face,faceID=fr.labels_for_training_data('C:/Users/a/Desktop/face rec/training')
face_recognizer=fr.train_classifier(face,faceID)
name={0:"karan",1:"priyal"}

for face in face_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(text_img,predicted_name,x,y)
resized_image=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial",resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

    

