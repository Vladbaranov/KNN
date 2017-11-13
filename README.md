# KNN
                                Определение класса, ближайших соседей, по обучающей выборке

Для определения класса по обучающей выборке на программном обеспечении Python необходимо подключить библиотеку numpy, matplotlib.pyplot и sklearn, и изъять нужные пакеты. 

`
print ( __ doc __ )

import  numpy as np

import matplotlib.pyplot as  pit

from matplotlib.colors import  ListedColormap

from sklearn import datasets

from sklearn.neighbors import NearestCentroid
`

После подготовки нужных пакетов вызываем функцию ирисов Фишеров и задаем количество ближайших соседей, размер шага, и координаты Х и У из функции Фишера.  Так же придаем цвет для Ирисов Фишера и областям, на которых находятся  ирисы Фишера.

`
n_neighbors = 20

iris = datasets.load_iris()

X = iris.data[:,   :2] у = iris.target

h =  .02    # размер шага

cmap_light  =  ListedColormap(['#FFAAAA',   'tfAAFFAA',   'ftAAAAFF'])  # Цвет облости 

cmap_bold =  ListedColormap(['#FF0000',   *#00FF00',   '#0000FF'])    # Цвет Ирисов Фишера
`

Следующим шагом идет написание цикла, который позволяет находить значения координат Х и У . После нахождения координат находим переменную Z для изображения ирисов Фишера.
`

for shrinkage in  [None,   .2]:

clf = NearestCentroid(shrink_threshold=shrinkage)

clf.fit(X,  y)

y_pred| = elf.predict(X)

print(shrinkage,  np.mean(y == y_pred))

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + l y_min, y_max = X[:, l].min() - 1, X[:, l].max() + 1 xx,  yy = 
np.meshgrid(np.arange(x_min,  x_max,  h),

np.arange(y_min,  y_max,  h)) Z = elf.predict(np.c_[xx.ravel(),  yy.ravel()])

Z =  Z.reshapefxx.shape)
`

После нахождения  координат составляем области, к которым присваиваются определенным цветом. Так же даем название графику.
`

pit.figure()

plt.pcolormesh(xx,  yy,  Z,  cmap=cmap_light)

pit.scatter(X[:,  0],  X[:,  1],  c=y,  cmap=cmap_bold>

edgecolor='b',  s=20) pit.title("3-Class classification  (shrink_threshold=%r)"

% shrinkage) plt.axis('tight')

pit.show()
`

После проверки работоспособности кода выходит график, на котором показаны расположение точек и принадлежность  к определенному виду  ирисов.


![](https://raw.githubusercontent.com/Vladbaranov/KNN/master/5.1.png)
