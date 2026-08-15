import cv2

l = []

for i in range(4):
    cam = cv2.VideoCapture(i)

    if cam.isOpened():
        l.append(i)
        cam.release()
    
cam.release()
print(l)