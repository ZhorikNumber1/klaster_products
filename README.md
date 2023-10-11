# Кластеризация товаров по частоте изменений

Данный код позволяет выполнить кластеризацию товаров на основе частоты изменений доступного количества. Он использует алгоритм DBSCAN для определения кластеров и визуализирует результаты.

## Зависимости

Для запуска этого кода вам необходимо установить следующие библиотеки:

- pandas
- numpy
- scikit-learn
- matplotlib

Вы можете установить их с помощью пакетного менеджера pip, выполнив следующие команды:


pip install pandas numpy scikit-learn matplotlib

Если у вас установлен Anaconda, вы можете использовать такую команду:

conda install pandas numpy scikit-learn matplotlib

## Запуск
1. Скачайте файлы проекта.
2. Установите необходимые зависимости.
4. Добавье в папку проекта файл csv
5. Запустите main.py, используя Python.
7. Введи в терминал навание фалй полностью
8. Ожидайте выволнение


python main.py

5. Результаты кластеризации будут сохранены в файл "clustered_data.csv", где указаны URL каждого товара, его кластер и категория изменений.

## Визуализация
После выполнения кода откроется график, на котором точки представляют товары, раскрашенные в соответствии с их кластерами. Кластеры будут добавлены в виде меток к соответствующим точкам на графике.
По осями X и Y отложенна частота. 