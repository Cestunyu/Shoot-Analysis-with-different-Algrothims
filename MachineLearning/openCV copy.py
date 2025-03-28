import cv2
import numpy as np


def on_mouse(event, x, y, flag, param):
    global get_hoop
    global coor_x
    global coor_y
    new_frame = frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        coor_x = x
        coor_y = y
        cv2.rectangle(new_frame, (coor_x-half_length, coor_y-half_length), (coor_x+half_length, coor_y+half_length), (0, 0, 255), 2)
        cv2.imshow("hoop", new_frame)
        get_hoop = 1
        cv2.destroyWindow("hoop")


path = "../video/CA00013.1.mov"
video = cv2.VideoCapture(path)
half_length = 24
get_hoop = 0
coor_x = coor_y = 0
count = 0

imgs = np.zeros((50000, 2*half_length, 2*half_length))

while True:
    ret, frame = video.read()
    if not ret:
        break
    if not get_hoop:
        cv2.namedWindow("hoop")
        cv2.imshow("hoop", frame)
        cv2.setMouseCallback("hoop", on_mouse)
        cv2.waitKey(0)


    cv2.rectangle(frame, (coor_x-half_length, coor_y-half_length), (coor_x+half_length, coor_y+half_length), (0, 0, 255), 2)
    cv2.imshow("video", frame)
    hoop_img = cv2.cvtColor(frame[coor_y-half_length:coor_y+half_length, coor_x-half_length:coor_x+half_length], cv2.COLOR_BGR2GRAY)
    im_array = np.array(hoop_img)
    print(im_array)
    # cv2.imwrite(image + str(count) + ".png", im_array)
    imgs[count] = im_array
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
np.save("../label/image.npy", imgs)


