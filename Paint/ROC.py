import numpy
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=numpy.inf)
Recall = numpy.zeros((20000,1))
False_Alarm_Rate = numpy.zeros((20000,1))

y_pred=numpy.load("C:\\Users\\13266\\Desktop\\test1\\label\\train_2\\p_y_given_x.npy")
y=numpy.load("C:\\Users\\13266\\Desktop\\test1\\label\\train_2\\test_set_y.npy")
# print(y.shape)#9244
# print(y_pred.shape)
y=y.astype('int64')
print(y_pred.shape)
y_pred_positive=y_pred
y_pred_positive_sort=sorted(y_pred_positive)
# print(y_pred_positive_sort)
threshold=y_pred_positive_sort
print(len(threshold))# 5488

j=0
for i_threshold in threshold:
    y_pred_positive_bool=y_pred_positive<=i_threshold

    y_pred_positive_bool=y_pred_positive_bool.astype('int64')
    y_pred_positive_bool=numpy.array(y_pred_positive_bool)
    y=numpy.array(y)

    recall=0
    false_alarm_rate=0

    # print(y)
    # for i in range(20000):
    #     # print((y_pred_positive_bool[i], y[i]))
    #     if y_pred_positive_bool[i]==y[i]:
    #         if y[i]==1:
    #             recall+=1
    #     if y_pred_positive_bool[i]==1:
    #         if y[i]==0:
    #             # print(y_pred_positive_bool[i]==1 )
    #             false_alarm_rate+=1
    # print(recall)
    # print(sum(y))
    # print(false_alarm_rate)
    Recall[j]=numpy.sum(y_pred_positive_bool&y)/numpy.sum(y)
    False_Alarm_Rate[j]=numpy.sum(y_pred_positive_bool-y_pred_positive_bool&y)/(20000-numpy.sum(y))
    # print(Recall[j])
    # print(False_Alarm_Rate[j])
    j+=1
numpy.save("C:\\Users\\13266\\Desktop\\test1\\label\\train_2\\False_Alarm_Rate.npy", False_Alarm_Rate)
numpy.save("C:\\Users\\13266\\Desktop\\test1\\label\\train_2\\Recall.npy", Recall)
plt.plot(False_Alarm_Rate,Recall)
plt.show()



