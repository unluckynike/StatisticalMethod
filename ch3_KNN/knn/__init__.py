'''
@Project ：StatisticalMethod 
@File    ：__init__.py.py
@Author  ：hailin
@Date    ：2022/10/30 14:54 
@Info    :  k 近 邻 法 代码实现
'''

import numpy as np
from collections import Counter
from math import sqrt
from sklearn.datasets import \
    make_classification  # 使用sklearn.datasets中的make_classification生成数据集，包括数据和标签,该函数生成数据集是随机的，每次运行得到的都不一样


# 定义KNN模型
class KNN(object):
    def __init__(self, k):
        self.k = k
        self.data = None
        self.label = None

    def train(self, data, label):
        self.data = data
        self.label = label

    def predict(self, x):
        dis = [sqrt(np.sum((xtrain - x) ** 2)) for xtrain in self.data]  # p=2 欧式距离
        # print('dis:',dis)
        # 根据距离进行排序，这里返回的是相应的索引号
        nearest = np.argsort(dis) # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # print("nearest:",nearest)
        # 获取前k个标签topK_y
        topk_y = [self.label[i] for i in nearest[:self.k]]
        # 计算topK_y中每种标签的数量，votes是个字典
        votes = Counter(topk_y)
        return votes.most_common(1)[0][0]


def createdata(n):
    '''
      n_samples:生成样本的数量
      n_features=2:生成样本的特征数，特征数=n_informative（） + n_redundant + n_repeated
      n_informative：多信息特征的个数
      n_redundant：冗余信息，informative特征的随机线性组合
      n_clusters_per_class ：某一个类别是由几个cluster构成的
      make_calssification默认生成二分类的样本，x代表生成的样本空间（特征空间,y代表了生成的样本类别，使用1和0分别表示正例和反例
      y=[0 0 0 1 0 1 1 1... 1 0 0 1 1 0]
    '''
    x, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    return x, y

# 生成120个样本，前100个样本用作训练集，其余的用作测试集建立KNN分类器，K值取2，进行测试
data, label = createdata(120)
knncla = KNN(2)
knncla.train(data[:100], label[:100])
ty = [knncla.predict(x) for x in data[100:]]
y = [i for i in label[100:]]
print('y:\n',y)
print('ty:\n',ty)
