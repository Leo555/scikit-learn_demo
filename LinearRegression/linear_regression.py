import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def runplt():
    plt.figure()
    plt.title(u'Height-Weight')
    plt.xlabel(u'Height')
    plt.ylabel(u'Weight')
    plt.axis([150, 190, 40, 120])
    plt.grid(True)
    return plt

plt = runplt()
X = [[155], [157], [166], [177], [187]]
y = [[55], [60], [63], [70], [79]]
plt.plot(X, y, 'k.')

plt.show()

# 创建并拟合模型
model = LinearRegression()
model.fit(X, y)


print('预测身高180同学的体重：%.2f' % model.predict(np.array([180]).reshape(-1, 1))[0])

# 残差预测值
y2 = model.predict(X)
plt.plot(X, y, 'k.')
plt.plot(X, y2, 'g-')

for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], y2[idx]], 'r-')

plt.show()