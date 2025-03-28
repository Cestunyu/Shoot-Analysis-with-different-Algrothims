import numpy
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=numpy.inf)

False_Alarm_Rate_bs_100=numpy.load("False_Alarm_Rate_bs_500_lr_0.05_nS_1000.npy")
False_Alarm_Rate_bs_200=numpy.load("False_Alarm_Rate_bs_500_lr_0.05_nS_2000.npy")
False_Alarm_Rate_bs_300=numpy.load("False_Alarm_Rate_bs_500_lr_0.05_nS_4000.npy")
False_Alarm_Rate_bs_400=numpy.load("False_Alarm_Rate_bs_500_lr_0.05_nS_8000.npy")
False_Alarm_Rate_bs_500=numpy.load("False_Alarm_Rate_bs_500_lr_0.05_nS_32000.npy")
False_Alarm_Rate_bs_600=numpy.load("False_Alarm_Rate_bs_500_lr_0.05_nS_48000.npy")

Recall_bs_100=numpy.load("Recall_bs_500_lr_0.05_nS_1000.npy")
Recall_bs_200=numpy.load("Recall_bs_500_lr_0.05_nS_2000.npy")
Recall_bs_300=numpy.load("Recall_bs_500_lr_0.05_nS_4000.npy")
Recall_bs_400=numpy.load("Recall_bs_500_lr_0.05_nS_8000.npy")
Recall_bs_500=numpy.load("Recall_bs_500_lr_0.05_nS_32000.npy")
Recall_bs_600=numpy.load("Recall_bs_500_lr_0.05_nS_48000.npy")

plt.title('Sample Number')
#plt.xscale('log')
plot_1=plt.plot(False_Alarm_Rate_bs_100,Recall_bs_100,'-g',label='Sample Number = 1000')
plot_2=plt.plot(False_Alarm_Rate_bs_200,Recall_bs_200,'-b',label='Sample Number = 2000')
plot_3=plt.plot(False_Alarm_Rate_bs_300,Recall_bs_300,'-r',label='Sample Number = 4000')
plot_4=plt.plot(False_Alarm_Rate_bs_400,Recall_bs_400,'cornflowerblue',label='Sample Number = 8000')
plot_5=plt.plot(False_Alarm_Rate_bs_500,Recall_bs_500,'yellow',label='Sample Number = 32000')
plot_6=plt.plot(False_Alarm_Rate_bs_600,Recall_bs_600,'gray',label='Sample Number = 48000')

plt.legend()
plt.xlabel("False_Alarm_Rate")
plt.ylabel("Recall")
plt.show()

# plt.figure("ROC Curve", figsize=(10, 6), dpi=80)
# colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'gray', 'blue', 'orange', 'black']
#     for i, color in zip(range(roc_all.shape[0]), colors):
#         plt.plot(roc_all[i][0], roc_all[i][1], color=color, label=roc_all[i][2], linewidth=2)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title("ROC Curve", fontsize=16)
# plt.legend(loc="lower right", fontsize=10)
# plt.show()


# plt.xlim(1e-5,1e-2)
