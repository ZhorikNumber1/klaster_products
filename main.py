import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
a = input()
# Загрузка данных
data = pd.read_csv(a)

# Конвертация даты и времени в нужный формат
data['created'] = pd.to_datetime(data['created'])

# Группировка данных по url и подсчет разницы в доступном количестве и временной разницы
data["available_quantity_diff"] = data.groupby("url")["available_quantity"].diff().abs()
data["time_diff"] = data.groupby("url")["created"].diff().abs().dt.total_seconds()
data["frequency"] = data["available_quantity_diff"] / data["time_diff"]

# Удаление возможных пропущенных значений
data.dropna(subset=["frequency"], inplace=True)

# Выделение данных для кластеризации
X = data[["frequency"]].values

# Нормализация данных
X = StandardScaler().fit_transform(X)

# Параметры для метода DмSCAN можно подстроить под вашу задачу
dbscan = DBSCAN(eps=0.5, min_samples=5)
data["cluster"] = dbscan.fit_predict(X)

# Назначение скоростей изменений по категориям
thresholds = [0.1, 1]  # Пороги изменений
labels = ["None", "slow", "fast"]
data["change_rate"] = pd.cut(data["frequency"], bins=[-np.inf] + thresholds + [np.inf], labels=labels)

# Сохранение результатов
data[["url", "cluster", "change_rate"]].to_csv("clustered_data.csv", index=False)

# Визуализация кластеров
plt.scatter(X[:,0], X[:,0], c=data["cluster"])
plt.title("Кластеризация данных")

# Добавление меток к точкам
for i in range(len(X)):
    plt.annotate(data['cluster'].iloc[i], (X[i, 0], X[i, 0]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
