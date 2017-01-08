import matplotlib.pyplot as plt
def runplt():
    plt.figure()
    plt.title(u'Height-Weight')
    plt.xlabel(u'Height')
    plt.ylabel(u'Weight')
    plt.axis([150, 190, 40, 90])
    plt.grid(True)
    return plt

plt = runplt()
X = [[155], [157], [166], [177], [187]]
y = [[55], [60], [63], [70], [79]]
plt.plot(X, y, 'k.')
plt.show()