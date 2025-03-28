from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy

import matplotlib.pyplot as plt

global I,Recall,False_Alarm_Rate,loss


Recall = numpy.zeros((1000, 1))
False_Alarm_Rate = numpy.zeros((1000, 1))
loss=numpy.zeros((1000,1))
I=0
J=0
class LogisticRegression(object):
    def __init__(self, n_in, n_out,batch_size,learning_rate,dataset,is_line_search=0,is_weight_decay=0):

        self.is_weight_decay=is_weight_decay

        self.is_line_search=is_line_search

        self.W = numpy.zeros((n_in, n_out))

        self.b = numpy.zeros((n_out, ))

        self.params = [self.W, self.b]

        self.batch_size=batch_size

        self.n_out=n_out

        self.n_in=n_in

        self.learning_rate=learning_rate

        self.dataset=dataset


    def load_data(self):
        # data_dir, data_file = os.path.split(dataset)
        # if data_dir == "" and not os.path.isfile(dataset):
        #     # Check if dataset is in the data directory.
        #     new_path = os.path.join(
        #         os.path.split(__file__)[0],
        #         "..",
        #         "data",
        #         dataset
        #     )
        #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        #         dataset = new_path
        #
        # if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        #     from six.moves import urllib
        #     origin = (
        #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        #     )
        #     print('Downloading data from %s' % origin)
        #     urllib.request.urlretrieve(origin, dataset)
        #
        # print('... loading data')
        #
        # # Load the dataset
        # with gzip.open(dataset, 'rb') as f:
        #     try:
        #         train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        #     except:
        #         train_set, valid_set, test_set = pickle.load(f)

        # test_set_x, test_set_y = test_set
        # valid_set_x, valid_set_y =valid_set
        # train_set_x, train_set_y = train_set

        self.train_set_x=numpy.load(self.dataset+"../data/Best_data/train/x_1.npy")
        self.valid_set_x=numpy.load(self.dataset+"../data/Best_data/valid/x_1.npy")
        self.test_set_x = numpy.load(self.dataset+"../data/Best_data/test/x_1.npy")

        # print(self.test_set_x[0])

        self.train_set_y=numpy.load(self.dataset+"../data/Best_data/train/y_1.npy")
        self.valid_set_y=numpy.load(self.dataset+"../data/Best_data/valid/y_1.npy")
        self.test_set_y=numpy.load(self.dataset+"../data/Best_data/test/y_1.npy")

        self.train_set_y=self.train_set_y.astype('int64')
        self.valid_set_y=self.valid_set_y.astype('int64')
        self.test_set_y=self.test_set_y.astype('int64')
        # print(test_set_x.shape)
        # print(valid_set_x.shape)
        # print(train_set_x.shape)

        # self.datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
        #         (test_set_x, test_set_y)]

        # self.train_set_x, self.train_set_y = self.datasets[0]
        # self.valid_set_x, self.valid_set_y = self.datasets[1]
        # self.test_set_x, self.test_set_y = self.datasets[2]

        self.n_train_batches = self.train_set_x.shape[0] // self.batch_size
        self.n_valid_batches = self.valid_set_x.shape[0] // self.batch_size
        self.n_test_batches = self.test_set_x.shape[0] // self.batch_size


    def compute_p_y_given_x(self,input,j=-1,test=-1):
        if j==-1:
            # print("j==-1")
            # print(self.W)
            # print(numpy.dot(input, self.W) + self.b)
            self.exp_x_multiply_W_plus_b = numpy.exp(numpy.dot(input, self.W) + self.b)
        else:
            print("j!=-1")
            # print(self.exp_x_multiply_W_plus_b)
            print(self.W)
            # print(input)
            print(numpy.dot(input,self.W[:,j])+self.b[j])
            xx=numpy.exp(numpy.dot(input,self.W[:,j])+self.b[j])

            self.exp_x_multiply_W_plus_b[:,j]=xx[:]
        sigma=numpy.sum(self.exp_x_multiply_W_plus_b,axis=1)
        self.p_y_given_x=self.exp_x_multiply_W_plus_b/sigma.reshape(sigma.shape[0],1)
        if test==1:
            print(self.p_y_given_x)
            print("xxx")
            numpy.save(self.dataset+"p_y_given_x.npy",self.p_y_given_x)



    def compute_y_pred(self,input,test=-1):
         self.compute_p_y_given_x(input,test=test)
         self.y_pred=numpy.argmax(self.p_y_given_x, axis=1)
    #这里返回的是下标，也就是属于哪一类



    def test_model(self,index):
        x = self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        y = self.test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        self.compute_y_pred(x,test=1)
        return self.errors(y)



    def validate_model(self,index):
        x=self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        y=self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        self.compute_y_pred(x)
        return self.errors(y)



    def train_model(self,index):
        x = self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        y = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        self.compute_p_y_given_x(x)
        self.grad_W_b(index)
        self.update_W_b(index)
        return self.negative_log_likelihood(y)



    def grad_W_b(self,index):
        x = self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        y = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]

        y_is_j=(y.reshape(y.shape[0],1)==numpy.array(numpy.arange(self.n_out),dtype=int))
        coef=y_is_j-self.p_y_given_x


        if self.is_weight_decay:
            self.delta_W=(-1.0*numpy.dot(coef.transpose(),x)/y.shape[0]).transpose()+self.lamda*self.W
            self.delta_b=-1.0*numpy.mean(coef,axis=0)+self.lamda*self.b
        else:
            self.delta_W=(-1.0*numpy.dot(coef.transpose(),x)/y.shape[0]).transpose()
            self.delta_b = -1.0 * numpy.mean(coef, axis=0)


    def update_W_b(self,index):
        if self.is_line_search:
            self.wolfe_line_search(index)
        else:
            self.W -= self.learning_rate * self.delta_W
            self.b -= self.learning_rate * self.delta_b

    def wolfe_line_search(self,index):
        i=0
        c=0.5
        tau=0.5
        x = self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        y = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        print(self.delta_W)
        slope=(self.delta_W**2).sum(axis=0)
        while i<self.n_out:
            t_learning_rate=0.0001
            oriLoss=self.negative_log_likelihood(y)
            print(self.W[:,i])
            self.W[:,i]-=t_learning_rate*self.delta_W[:,i]
            print(self.W[:, i])
            prev_learning_rate=t_learning_rate
            while 1:
                tt=c*t_learning_rate*slope[i]
                self.compute_p_y_given_x(x,j=i)
                currLoss=self.negative_log_likelihood(y)
                if currLoss<=oriLoss-tt:
                    break
                else:
                    t_learning_rate*=tau
                    if t_learning_rate<self.learning_rate:
                        t_learning_rate=self.learning_rate
                        self.W[:,i]+=(prev_learning_rate-t_learning_rate)*self.delta_W[:,i]
                        self.compute_p_y_given_x(x,j=i)
                        break
                self.W[:,i]+=(prev_learning_rate-t_learning_rate)*self.delta_W[:,i]
                prev_learning_rate=t_learning_rate
        i+=1

    def negative_log_likelihood(self, y):
        # print(self.p_y_given_x)
        # print(y)
        # print(numpy.log(self.p_y_given_x))
        # print(-numpy.mean(numpy.log(self.p_y_given_x)[numpy.arange(y.shape[0]), y]))
        return -numpy.mean(numpy.log(self.p_y_given_x)[numpy.arange(y.shape[0]), y])


    # def testerrors(self,y):
    #     global I
    #     if y.ndim != self.y_pred.ndim:
    #         raise TypeError(
    #             'y should have the same shape as self.y_pred',
    #             ('y', y.type, 'y_pred', self.y_pred.type)
    #         )
    #     y=numpy.array(y)
    #     pred=numpy.array(self.y_pred)
    #
    #     # print(y&pred)
    #     # Recall[I]=sum(y&pred)/sum(y)
    #     # False_Alarm_Rate[I]=sum(pred-pred&y)/(numpy.size(y)-sum(y))
    #     # I+=1
    #     return numpy.mean(self.y_pred != y),sum(y&pred),sum(pred-pred&y)

    def recall(self,index):
        pred = numpy.array(self.y_pred)
        y = self.test_set_y[index * self.batch_size: (index + 1) * self.batch_size]#cpmpte_y已经算了
        # print("Y:")
        # print(sum(y))
        # print("pred:")
        # print(sum(pred))
        return sum(y&pred)

    def false_alarm_rate(self,index):
        pred = numpy.array(self.y_pred)
        y = self.test_set_y[index * self.batch_size: (index + 1) * self.batch_size]  # cpmpte_y已经算了
        # print(y.shape)
        # print(pred.shape)
        return sum(pred-pred&y)

    def errors(self,y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred',self.y_pred.type)
            )
        # print(y)
        # if y.dtype.startswith('int'):
        return numpy.mean(self.y_pred!=y)
        # else:
        #     raise NotImplementedError()
    def draw(self):
        Y= Recall[0:I+1]
        X= False_Alarm_Rate[0:I+1]
        numpy.save(self.dataset+"False_Alarm_Rate.npy",X)
        numpy.save(self.dataset + "Recall.npy", Y)
        plt.plot(X, Y)
        # X=numpy.load(self.dataset+"loss.npy")
        plt.show()


def sgd_optimization_mnist(learning_rate=0.13,
                           n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600,
                           n_in=28 * 28,
                           n_out=10,
                           is_line_search=0,
                           is_weight_decay=0):
    print('... building the model')
    global I
    classifier = LogisticRegression(n_in=n_in,
                                    n_out=n_out,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    is_line_search=is_line_search,
                                    is_weight_decay=is_weight_decay,
                                    dataset=dataset)

    classifier.load_data()

    print('... training the model')
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(classifier.n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(classifier.n_train_batches):#每次循环完成训练所有样本
            minibatch_avg_cost = classifier.train_model(minibatch_index)  #这里调用train_model将会进行一次梯度下降

            iter = (epoch - 1) * classifier.n_train_batches + minibatch_index#表示相对所有来说训练了多少个batch了

            if (iter + 1) % validation_frequency == 0:#每frequency个batch验证一次，这是不确定的一个数
                # 这里为n_tarin_batches就相当于所有样本训练完检测一次，此时恰好完成一个epoch
                #每frequency验证一下，而这里frequency取的是n_train_batches,因此每次为83/83
                #
                validation_losses = [classifier.validate_model(i)
                                     for i in range(classifier.n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, the loss is %f ,validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        classifier.n_train_batches,
                        minibatch_avg_cost,
                        this_validation_loss * 100.
                    )
                )
                loss[epoch-1]=minibatch_avg_cost

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses=[classifier.test_model(i)
                                   for i in range(classifier.n_test_batches)]
                    # recall=[classifier.recall(i)
                    #                for i in range(classifier.n_test_batches)]
                    # false_alarm_rate=[classifier.recall(i)
                    #                for i in range(classifier.n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    # print(sum(recall))
                    # print(sum(false_alarm_rate))
                    # Recall[I]=sum(recall)/sum(classifier.test_set_y)
                    # False_Alarm_Rate[I]=sum(false_alarm_rate)/(numpy.size(classifier.test_set_y)-sum(classifier.test_set_y))
                    # I+=1
                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            classifier.n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model_train_1_W_b.pkl', 'wb') as f:
                        pickle.dump([classifier.W,classifier.b,
                                     classifier.n_in, classifier.n_out,
                                     classifier.batch_size,classifier.learning_rate,
                                     classifier.dataset,classifier.is_line_search,
                                     classifier.is_weight_decay], f)

            if patience <= iter:
                done_looping = True
                break
    # classifier.draw()
    numpy.save(dataset+"loss.npy",loss[0:epoch])
    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


# def predict():
#     """
#     An example of how to load a trained model and use it
#     to predict labels.
#     """
#
#     # load the saved model
#     classifier = pickle.load(open('best_model.pkl'))
#
#     # compile a predictor function
#     predict_model = theano.function(
#         inputs=[classifier.input],
#         outputs=classifier.y_pred)
#
#     # We can test it on some examples from test test
#     dataset='mnist.pkl.gz'
#     datasets = load_data(dataset)
#     test_set_x, test_set_y = datasets[2]
#     test_set_x = test_set_x.get_value()
#
#     predicted_values = predict_model(test_set_x[:10])
#     print("Predicted values for the first 10 examples in test set:")
#     print(predicted_values)


print("--------------------")
print("Read the configure file:")
fp = open('configure.txt', 'r')
strConfig = fp.readlines()
# init
Learning_rate = '',
N_epochs = '',
Dataset = '',
Batch_size = '',
N_in = '',
N_out = '',
Is_line_search = '',
Is_weight_decay = ''
# read the strConfig

for line in strConfig :
    line = line.strip('')
    #print(line)
    if line.find('is_line_search') >= 0:
        Is_line_search =(int)(line.split('=')[1].strip())
        print(Is_line_search)
    if line.find('is_weight_decay') >= 0:
        Is_weight_decay =(int)(line.split('=')[1].strip())
        print(Is_weight_decay)
    if line.find('n_out') >= 0:
        N_out = (int)(line.split('=')[1].strip())
        print(N_out)
    if line.find('n_in') >= 0:
        N_in = (int)(line.split('=')[1].strip())
        print(N_in)
    if line.find('batch_size') >= 0:
        Batch_size = (int)(line.split('=')[1].strip())
        print(Batch_size)
    if line.find('dataset') >= 0:
        Dataset = line.split('=')[1].strip()
        print(Dataset)
    if line.find('n_epochs') >= 0:
        N_epochs = (int)(line.split('=')[1].strip())
        print(N_epochs)
    if line.find('learning_rate') >= 0:
        Learning_rate = (float)(line.split('=')[1].strip())
        print(Learning_rate)

sgd_optimization_mnist(learning_rate=Learning_rate,
                       n_epochs=N_epochs,
                       dataset=Dataset,
                       batch_size=Batch_size,
                       n_in=N_in,
                       n_out=N_out,
                       is_line_search=Is_line_search,
                       is_weight_decay=Is_weight_decay)


