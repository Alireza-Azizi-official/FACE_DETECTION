# import libraries to show and modify images.
import cv2
import matplotlib.pyplot as plt

# set a variable to import the specific image.
image_path = 'input_image.jpg'

# use cv2 to read the image.
img = cv2.imread(image_path)

# each pic has 3 dimensions we don't need one of them which is colors so we change it to gray also it helps to recognize better.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect the faces in the picture.
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=3, minSize=(100, 100))

# make a box to show around the face.
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# change the format from bgr to rgb.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# crop the detected faces from the image and save it in a folder.
i=0
for x,y,w,h in face:
    crop_faces=img[y:y+h,x:x+w]
    target_file="face_detection"+str(i)+".jpg"
    cv2.imwrite(target_file,crop_faces,)
    i=i+1
    
# choose the size of the window and show the pictures of detected faces.
plt.figure(figsize=(5,5))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

