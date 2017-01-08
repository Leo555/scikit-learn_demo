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

polynomial_featurizer = PolynomialFeatures(degree=3)
x_train_polynomial = polynomial_featurizer.fit_transform(x)
x_test_polynomial = polynomial_featurizer.transform(x_test)
model_polynomial = LinearRegression()
model_polynomial.fit(x_train_polynomial, y)
xx_polynomial = polynomial_featurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, model_polynomial.predict(xx_polynomial), 'r-')
plt.show()

print('1 r-squared', model.score(x_test, y_test))
print('2 r-squared', model_polynomial.score(x_test_polynomial, y_test))
