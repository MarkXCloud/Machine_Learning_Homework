from Load_data import my_load_iris
import numpy as np
from SVM import Support_Vector_Machine
from Horizontal_Classify import Horizontal_Classifier

class Adaboost():
    def __init__(self, num_classifier):
        """
        :param num_classifyer: the number of weak classifier
        """
        self.x_train, self.y_train = None, None
        self.num_classifyer = num_classifier
        self.num_data = 0
        self.w = None
        self.theta = None
        self.models = []

    def load_data(self, data, label):
        self.x_train = data
        self.y_train = label

    def init_params(self):
        """
        Since some parameters are related to the dataset, so I didn't put these param init in the __init__()
        but to separately initiate.
        :return:
        """
        self.num_data = self.x_train.shape[0]
        self.w = np.ones(self.num_data) / self.num_data
        self.theta = np.zeros(self.num_classifyer)

    def load_weak_classifier(self, weak_classifier='Line'):
        """
        Now we have two choices:SVM and single line.
        SVM performs better, and need fewer num_classifier.
        Single line performs dummy, but as the num_classifier increase, adaboost will remedy them.
        :param weak_classifier:
        :return:
        """
        for i in range(self.num_classifyer):
            if weak_classifier == 'SVM':
                self.models.append(Support_Vector_Machine())
            elif weak_classifier == 'Line':
                self.models.append(Horizontal_Classifier())

    def fit(self):
        self.init_params()
        self.load_weak_classifier()
        for m in range(self.num_classifyer):
            self.models[m].load_data(self.x_train, self.y_train)
            self.models[m].fit(epoch=10)
            pred_train = self.models[m].predict(self.x_train)
            miss = [int(x) for x in (pred_train != self.y_train)]
            error = np.dot(self.w, miss)
            self.theta[m] = 0.5 * np.log((1 - error) / (error))

            for i in range(self.num_data):
                self.w[i] = self.w[i] * np.exp(-self.theta[m] * self.y_train[i] * pred_train[i])

            Zm =np.sum(self.w * np.exp(-self.theta[m] * self.y_train * pred_train))
            self.w /= Zm

    def predict(self, x_test):
        # 最终的预测
        predict = np.dot(self.theta, [self.models[m].predict(x_test) for m in range(self.num_classifyer)])
        print(predict)


if __name__ == '__main__':
    adaboost = Adaboost(num_classifier=20)
    adaboost.load_data(*my_load_iris())
    adaboost.fit()
    adaboost.predict([[5.7, 2.8],[4.9,3]])
