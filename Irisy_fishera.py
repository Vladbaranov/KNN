print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 20 # Задаем количество ближайших соседей

iris = datasets.load_iris()  # Извлекаем данные для работы с выборкой Ирисов

# Сохраняем первые два параметра выборки
X = iris.data[:, :2]
y = iris.target

h = .2  # Задаем размер шага


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])# цвет области
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # цвет Ирисов Фишера

for weights in ['uniform', 'distance']:
    # Создаем классификатор ближайших соседей
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Задаем границу решения. Для этого определим цвет каждой точке [x_min, x_max] x [y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Выводим полученный результат
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Выводим исходную выборку
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
