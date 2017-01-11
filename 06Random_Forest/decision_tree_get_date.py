from numpy import genfromtxt
from sklearn import tree

# 加载数据
dataset = genfromtxt('data.csv', delimiter=",")
x = dataset[1:, 0:4]
y = dataset[1:, 4]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

# 預測
print(clf.predict([[35, 0, 100, 1]]))