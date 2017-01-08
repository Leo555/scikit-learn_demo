import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def runplt():
    plt.figure()
    plt.title(u'Height-Weight')
    plt.xlabel(u'Height')
    plt.ylabel(u'Weight')
    plt.axis([150, 190, 40, 90])
    plt.grid(True)
    return plt


plt = runplt()
x = [[155], [157], [166], [177], [187]]
y = [[55], [60], [63], [70], [79]]
plt.plot(x, y, 'k.')

plt.show()

# 创建并拟合模型
model = LinearRegression()
model.fit(x, y)

print('预测身高180同学的体重：%.2f' % model.predict(np.array([180]).reshape(-1, 1))[0])

# 残差预测值
y2 = model.predict(x)
plt.plot(x, y, 'k.')
plt.plot(x, y2, 'g-')

for idx, x in enumerate(x):
    plt.plot([x, x], [y[idx], y2[idx]], 'r-')

plt.show()

import numpy as np

print('残差平方和: %.2f' % np.mean((model.predict(x) - y) ** 2))

## 测试集
x_test = [[156], [163], [166], [170], [188]]
y_test = [[56], [63], [63], [72], [80]]
print('R方： ', model.score(x_test, y_test))
