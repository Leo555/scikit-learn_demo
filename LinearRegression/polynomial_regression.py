import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def runplt():
    plt.figure()
    plt.title(u'Height-Weight')
    plt.xlabel(u'Height')
    plt.ylabel(u'Weight')
    plt.axis([150, 190, 40, 90])
    plt.grid(True)
    return plt


x = [[155], [157], [166], [177], [187]]
y = [[55], [60], [63], [70], [79]]
x_test = [[156], [163], [166], [170], [188]]
y_test = [[56], [63], [63], [72], [80]]
# 建立线性回归，并用训练的模型绘图
model = LinearRegression()
model.fit(x, y)
xx = np.linspace(150, 190, 100)
yy = model.predict(xx.reshape(xx.shape[0], 1))
plt = runplt()
plt.plot(x, y, 'k.')
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=5)
x_train_quadratic = quadratic_featurizer.fit_transform(x)
x_test_quadratic = quadratic_featurizer.transform(x_test)
model_quadratic = LinearRegression()
model_quadratic.fit(x_train_quadratic, y)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, model_quadratic.predict(xx_quadratic), 'r-')
plt.show()

print(x)
print(x_train_quadratic)
print(x_test)
print(x_test_quadratic)
print('1 r-squared', model.score(x_test, y_test))
print('2 r-squared', model_quadratic.score(x_test_quadratic, y_test))
