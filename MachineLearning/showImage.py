# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

#
# plt.figure()
# plt.subplot(1, 2, 1)
#

# for i in range(1, 26):
#     plt.subplot(5, 5, i)
#     plt.imshow(images[i+30], cmap="gray")
# plt.axis("off")
# plt.tight_layout()
# plt.show()


# init npy 1
# np.save("../array/", label_y)
images = np.load("../array/CA00013_1_z.npy")
label_y = np.load("../array/label_y.npy")
# the label of every frame

print(images.shape)
plt.figure()
# ----------------------------laoding the txt. text---------------------
print("Read the moment file:")
fp = open('../array/Moment.txt', 'r')
strMoment = fp.readlines()
Moment = [0] * 400  # the array to store positive moments
numMoment = 1  # the number of moments

for line in strMoment:
    line = line.strip('')
    Moment[numMoment] = int(line)
    # print(numMoment, Moment[numMoment])
    numMoment += 1

AllnumMoment = numMoment
numMoment = 229
# init

# ------------------------------loading------------------------


while Moment[numMoment] != 0:
    print("The picture number:", numMoment,numMoment/AllnumMoment, "The second:", Moment[numMoment])
    print("---------------------------------------------")
    for i in range(1, 26):
        # print(i)
        plt.subplot(5, 5, i)
        plt.imshow(images[i + Moment[numMoment] * 25])
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # for i in range(1, 26):
    #     print(i)
    #     if i % 5 == 0:
    #         print("Row:", (int)(i / 5) + 1, "Column:", 5)
    #     else:
    #         print("Row:", (int)(i / 5) + 1, "Column:", i%5)
    #     # the location of the picture
    head = input("Enter the head: ")
    if head != ' ':
        end = input("Enter the end: ")
        for i in range(int(head),int(end)):
         label_y[i+ (Moment[numMoment]) * 25] = 1

        # plt.imshow(images[i + Moment[numMoment] * 25], cmap="gray")
        # display the bigger image
        # plt.show()
    # if str == 'p':  # positive
    #     label_y[i + (Moment[numMoment]) * 25] = 1
    #     plt.imshow(images[i + Moment[numMoment] * 25], cmap="gray")
    #     # display the bigger image
    #     plt.show()

    numMoment += 1
    np.save("../array/label_y.npy", label_y)


