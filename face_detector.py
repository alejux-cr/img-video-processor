import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("news.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,
scaleFactor=1.1, 
minNeighbors=5)  # scaleFactor 0.5 is the scale it starts to break the img to search for faces  || minNeighbors how many neighbors to search around the window, 5 is acceptable

print(faces)
print(type(faces))

for x,y,width,height in faces:
    img = cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),3)


resized = cv2.resize(img, (int(img.shape[1]/3),int(img.shape[0]/3)))
cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

