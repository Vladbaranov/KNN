# KNN
                               
Для определения класса по обучающей выборке на языке Python, необходимо подключить библиотеки: numpy, matplotlib.pyplot и sklearn, и изъять нужные пакеты.
```python
print ( __ doc __ )
import  numpy as np
import matplotlib.pyplot as  pit
from matplotlib.colors import  ListedColormap
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
```
После подготовки нужных пакетов выгружаем необходимые данные из пакета datasets, т.е. обучающую выборку и задаем количество ближайших соседей, размер шага, и находим координаты Х и У из пакета datasets.  Так же присваиваем цвет объектам выборки   и  областям, на которых находятся  эти объекты.
```python
n_neighbors = 20
iris = datasets.load_iris()
X = iris.data[:,   :2] у = iris.target
h =  .02    # размер шага
cmap_light  =  ListedColormap(['#FFAAAA',   'tfAAFFAA',   'ftAAAAFF'])  # Цвет области 
cmap_bold =  ListedColormap(['#FF0000',   '#00FF00',   '#0000FF'])    # Цвет Ирисов Фишера
```

Следующим шагом идет написание цикла, который позволяет находить значения координат Х и У . После нахождения координат находим переменную Z для изображения класса по обучающей выборке.
```python
for shrinkage in  [None,   .2]:
clf = NearestCentroid(shrink_threshold=shrinkage)
clf.fit(X,  y)
y_pred| = elf.predict(X)
print(shrinkage,  np.mean(y == y_pred))
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + l y_min, y_max = X[:, l].min() - 1, X[:, l].max() + 1 xx,  yy = 
np.meshgrid(np.arange(x_min,  x_max,  h),
np.arange(y_min,  y_max,  h)) Z = elf.predict(np.c_[xx.ravel(),  yy.ravel()])
Z =  Z.reshapefxx.shape)
```

После нахождения  координат составляем области, которым присваиваются определенные цвета. Так же даем название графику.

```python
pit.figure()
plt.pcolormesh(xx,  yy,  Z,  cmap=cmap_light)
pit.scatter(X[:,  0],  X[:,  1],  c=y,  cmap=cmap_bold>
edgecolor='b',  s=20) pit.title("3-Class classification  (shrink_threshold=%r)"
% shrinkage) plt.axis('tight')
pit.show()
```

После проверки работоспособности кода выходит график, на котором показаны расположение точек и принадлежность  к определенному виду  Ирисов.


![](https://raw.githubusercontent.com/Vladbaranov/KNN/master/5.1.png)
