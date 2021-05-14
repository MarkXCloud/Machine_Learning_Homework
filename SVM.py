import numpy as np
import matplotlib.pyplot as plt
from Load_data import my_load_iris
from SMO import smo, get_w


class Support_Vector_Machine():

    def __init__(self):
        self.data = None
        self.label = None
        self.alphas = None
        self.b = None

    def load_data(self, data, label):
        self.data, self.label = data, label

    def fit(self):
        """fit to classify by smo"""
        self.alphas, self.b = smo(self.data, self.label, 0.6, 40)

        # 分类数据点
        classified_pts = {'+1': [], '-1': []}
        for point, label in zip(self.data, self.label):
            if label == 1:
                classified_pts['+1'].append(point)
            else:
                classified_pts['-1'].append(point)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 绘制数据点
        for label, pts in classified_pts.items():
            pts = np.array(pts)
            ax.scatter(pts[:, 0], pts[:, 1], label=label)
        # 绘制分割线
        w = get_w(self.alphas, self.data, self.label)
        x1, _ = max(self.data, key=lambda x: x[0])
        x2, _ = min(self.data, key=lambda x: x[0])
        a1, a2 = w
        y1, y2 = (-self.b - a1 * x1) / a2, (-self.b - a1 * x2) / a2
        ax.plot([x1, x2], [y1, y2])
        # 绘制支持向量
        for i, alpha in enumerate(self.alphas):
            if abs(alpha) > 1e-3:
                x, y = self.data[i]
                ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                           linewidth=1.5, edgecolor='#AB3319')

    def predict(self, X):
        # SVM分类器函数 y = w^Tx + b
        result = []
        for x in X:
            # Kernel function vector.
            x = np.matrix(x).T
            data = np.matrix(self.data)
            ks = data * x
            # Predictive value.
            wx = np.matrix(self.alphas * self.label) * ks
            fx = wx + self.b
            result.append(1 if fx[0, 0] > 0 else -1)

        return np.array(result)

    def plot_origin_scatter(self, num=50):
        """
        Show the train data
        :param num: the split point of the two clases
        :return: scatter
        """
        plt.scatter(self.data[:num, 0], self.data[:num, 1], c='r')
        plt.scatter(self.data[num:, 0], self.data[num:, 1], c='b')


if __name__ == '__main__':
    SVM = Support_Vector_Machine()
    SVM.load_data(*my_load_iris())
    SVM.plot_origin_scatter()
    SVM.fit()
    print(SVM.predict([[5.7, 2.8]]))  # should be 1
    print(SVM.predict([[4.9, 3]]))  # should be -1
    plt.show()
