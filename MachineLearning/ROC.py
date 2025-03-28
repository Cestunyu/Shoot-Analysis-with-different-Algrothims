import numpy
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import numpy as np
# x:m*d w:k*d

mode_path = "p_y_given_x.npy"


def Hypothesis(w, x):
    score = np.dot(w, x.T)  # k*d*d*m = k*m
    score = np.exp(score - np.max(score, axis=0))  # 压缩，使其不溢出
    h = score / np.sum(score, axis=0)
    return h

def load_model(mode_path):
    weight = pickle.load(open(mode_path, 'rb'))
    return weight


def get_pro(weight, data):
    return Hypothesis(weight, data)


def get_roc(pred, y):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  # 从大到小排序
    index = np.argsort(pred)[::-1]  # 从大到小排序
    y_sort = y[index]
    tpr = []
    fpr = []
    thr = []
    for i, item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    return tpr, fpr


def plot_all_roc(roc_all):
    plt.figure("ROC Curve", figsize=(10, 6), dpi=80)
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'gray', 'blue', 'orange', 'black']
    for i, color in zip(range(roc_all.shape[0]), colors):
        plt.plot(roc_all[i][0], roc_all[i][1], color=color, label=roc_all[i][2], linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.show()


def get_all_model_roc(mode_path_all, test_x, test_y):
    mode = []
    roc_all = []
    for i in range(len(mode_path_all)):
        mode.append(load_model(mode_path_all[i]))
        pro = get_pro(mode[i], test_x)
        tpr, fpr = get_roc(pro[1], test_y)
        name = mode_path_all[i]
        param = [tpr, fpr, name]
        roc_all.append(param)
    roc_all = np.array(roc_all)
    plot_all_roc(roc_all)

# numpy.set_printoptions(threshold=numpy.inf)
# Recall = numpy.zeros((200000,1))
# False_Alarm_Rate = numpy.zeros((200000,1))
#
# y_pred=numpy.load("p_y_given_x.npy")
# y=numpy.load("../data/Best_data/test/y_1.npy")
# # print(y.shape)#9244
# # print(y_pred.shape)
# y=y.astype('int64')
# print(y_pred.shape)
# y_pred_positive=y_pred
# # y_pred_positive_sort=sorted(y_pred_positive)
# y_pred_positive_sort=y_pred_positive
# print(y_pred_positive_sort)
# threshold=y_pred_positive_sort
# print(len(threshold))# 5488
#
# j=0
#
# for i_threshold in threshold:
#     y_pred_positive_bool=y_pred_positive<=i_threshold
#
#     y_pred_positive_bool=y_pred_positive_bool.astype('int64')
#     y_pred_positive_bool=numpy.array(y_pred_positive_bool)
#     y=numpy.array(y)
#
#     recall=0
#     false_alarm_rate=0
#
#     # print(y)
#     for i in range(200000):
#         # print((y_pred_positive_bool[i], y[i]))
#         if y_pred_positive_bool[i]==y[i]:
#             if y[i]==1:
#                 recall+=1
#         if y_pred_positive_bool[i]==1:
#             if y[i]==0:
#                 # print(y_pred_positive_bool[i]==1 )
#                 false_alarm_rate+=1
#     Recall[j] = recall / numpy.sum(y)
#     False_Alarm_Rate[j] = false_alarm_rate / (200000 - numpy.sum(y))
#     # print(recall)
#     # print(sum(y))
#     # print(false_alarm_rate)
#     # Recall[j]=numpy.sum(y_pred_positive_bool&y)/numpy.sum(y)
#     # False_Alarm_Rate[j]=numpy.sum(y_pred_positive_bool-y_pred_positive_bool&y)/(20000-numpy.sum(y))
#     # print(Recall[j])
#     # print(False_Alarm_Rate[j])
#     j+=1
# numpy.save("False_Alarm_Rate.npy", False_Alarm_Rate)
# numpy.save("Recall.npy", Recall)
# print("F")
# plt.plot(False_Alarm_Rate,Recall)
# plt.show()
#
#
#
