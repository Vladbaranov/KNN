# KNN
                                Определение класса, ближайших соседей, по обучающей выборке

Для определения класса по обучающей выборке на программном обеспечении Python необходимо подключить библиотеку numpy, matplotlib.pyplot и sklearn, и изъять нужные пакеты. 

print (__ doc __)

import  numpy as np

import matplotlib.pyplot as  pit

from matplotlib.colors import  ListedColormap

from sklearn import datasets

from sklearn.neighbors import NearestCentroid

После подготовки нужных пакетов вызываем функцию ирисов Фишеров и задаем количество ближайших соседей, размер шага, и координаты Х и У из функции Фишера.  Так же придаем цвет для Ирисов Фишера и областям, на которых находятся  ирисы Фишера.

![](https://raw.githubusercontent.com/Vladbaranov/KNN/master/2.1.png)

Следующим шагом идет написание цикла, который позволяет находить значения координат Х и У . После нахождения координат находим переменную Z для изображения ирисов Фишера.

![](https://raw.githubusercontent.com/Vladbaranov/KNN/master/3.png)

После нахождения  координат составляем области, к которым присваиваются определенным цветом. Так же даем название графику.

![](https://raw.githubusercontent.com/Vladbaranov/KNN/master/4.png)

После проверки работоспособности кода выходит график, на котором показаны расположение точек и принадлежность  к определенному виду  ирисов.


![](https://raw.githubusercontent.com/Vladbaranov/KNN/master/5.1.png)
