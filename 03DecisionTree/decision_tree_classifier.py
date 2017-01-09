from numpy import genfromtxt
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image

# 加载数据
dataset = genfromtxt('data.csv', delimiter=",")
x = dataset[1:, 0:3]
y = dataset[1:, 3]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

# 預測
print(clf.predict([[0, 0, 50]]))

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("load.pdf")
Image(graph.create_png())

