#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from Load_data import my_load_iris
class Horizontal_Classifier():
    def __init__(self):
        """
        border:the location of a line vertical to the x axis
        data & label: our data to learn
        alpha:learning rate, should be small or the line will be swinging back and forth
        """
        self.border = 0
        self.data = None
        self.label = None
        self.alpha = 0.01

    def load_data(self,data,label):
        self.data = data
        self.label = label

    def predict(self,datas):
        result = []
        for data in datas:# simply judge the first dim
            if data[0] < self.border:#因为是水平分类器所以取了data的第一个维度，小的是-1
                result.append(-1)
            else:
                result.append(1)
        return np.array(result)

    def plot_results(self,num=50):
        print("border: ",self.border)
        plt.scatter(self.data[:num, 0], self.data[:num, 1], c='r')
        plt.scatter(self.data[num:, 0], self.data[num:, 1], c='b')
        plt.plot([self.border,self.border],[1,5],c='y')

    def fit(self,epoch=500):
        for _ in range(epoch):
            result = self.predict(self.data)
            miss = [int(x) for x in (result != self.label)]
            # 这里模仿了梯度下降的原理，对分错的样本针对性地进行更新分类界限
            self.border += -self.alpha * np.sum(np.dot(self.label,np.array(miss)))


if __name__ == '__main__':
    hc = Horizontal_Classifier()
    hc.load_data(*my_load_iris())
    hc.fit(epoch=15)
    hc.plot_results()
    print(hc.predict([[5.7, 2.8],[4.9,3]]))
    plt.show()