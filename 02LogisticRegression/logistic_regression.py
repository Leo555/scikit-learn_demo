import numpy as np
import urllib.request
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# 加载数据
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.request.urlopen(url)
# 把CSV文件转化为numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# 训练集和结果
X = dataset[0:700, 0:7]
y = dataset[0:700, 8]
# 数据归一化
normalized_X = preprocessing.normalize(X)
# 逻辑回归
model = LogisticRegression()

model.fit(normalized_X, y)

# 预测
expected = y
predicted = model.predict(normalized_X)

# 模型拟合概述
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
