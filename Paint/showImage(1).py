import numpy as np
import cv2
import matplotlib.pyplot as plt


images = np.load("C:\\Users\\13266\\Desktop\\test1\\label\\basketball_z.npy")
label_z = np.load("C:\\Users\\13266\\Desktop\\test1\\label\\label_z.npy")

sec=11
min=30
print(images.shape)
plt.figure()
for i in range(1, 26):
    print(i)
    plt.subplot(5, 5, i)
    plt.imshow(images[i+(min*60+sec)*25], cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

for i in range(1, 26):
    print((int)(i/5),i%5)
    str = input("Enter your input: ");
    if str == 'p':
        label_z[i+(min*60+sec)*25]=1
        plt.imshow(images[i + (min*60+sec)* 25], cmap="gray")
        plt.show()
    elif str == 'n':
       continue
np.save("C:\\Users\\13266\\Desktop\\test1\\label\\label_z.npy", label_z)