from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier

# 加载数据
dataset = genfromtxt('data.csv', delimiter=",")
x = dataset[1:, 0:4]
y = dataset[1:, 4]

clf = RandomForestClassifier(n_jobs=2, oob_score=True)
# 训练
clf = clf.fit(x, y)

# 預測
print(clf.predict([[33, 0, 100, 1]]))
print(clf.predict([[33, 1, 80, 0]]))
print(clf.predict_proba([[33, 0, 80, 0]]))
'''
参数说明：
1. max_features
随机森林允许单个决策树使用特征的最大数量。 Python为最大特征数提供了多个可选项。
(a) Auto/None ：简单地选取所有特征，每颗树都可以利用他们。这种情况下，每颗树都没有任何的限制。
(b) sqrt ：此选项是每颗子树可以利用总特征数的平方根个。 例如，如果变量（特征）的总数是100，所以每颗子树只能取其中的10个。“log2”是另一种相似类型的选项。
(c) 0.2：此选项允许每个随机森林的子树可以利用变量（特征）数的20％。如果想考察的特征x％的作用， 我们可以使用“0.X”的格式。

2. n_estimators
在利用最大投票数或平均值来预测之前，希望建立子树的数量。

3. min_sample_leaf
决策树最小样本叶片大小的重要性

4. n_jobs
这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。

5. verbose
是否显示工作进程

6. class_weight
各个label的权重

7. oob_score
这是一个随机森林交叉验证方法。这种方法只是简单的标记在每颗子树中用的观察数据。 然后对每一个观察样本找出一个最大投票得分，是由那些没有使用该观察样本进行训练的子树投票得到

8. warm_start
热启动，决定是否使用上次调用该类的结果然后增加新的
'''