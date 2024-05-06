import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error  # 均方误差MSE
from sklearn.metrics import mean_absolute_error  # 平方绝对误差MAE
from sklearn.metrics import r2_score  # R square R2

#plt.rcParams['font.sans-serif'] = ['KaiTi']
#plt.rcParams['axes.unicode_minus'] = False
# 图片像素
# plt.rcParams['savefig.dpi'] = 800
# 分辨率
#plt.rcParams['figure.dpi'] = 200

file_path = r'C:\Users\Administrator\Desktop\Four.csv'

data = pd.read_csv(file_path)
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

param_grid = {
    'n_estimators': [10, 50, 100, 200, 400],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt']
}

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# 训练模型
grid_search.fit(x_train, y_train)
# 预测
best_rf = grid_search.best_estimator_
y_predict = best_rf.predict(x_test)

MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
R2 = r2_score(y_test, y_predict)

print(np.mean(MSE))
print(np.mean(MAE))
print(np.mean(R2))

# train = svc.score(x_train, y_train)
# test = svc.score(x_test, y_test)
# print(train)
# print(test)

# """
# plt.figure(figsize=(10,8))
# plt.plot(range(1,len(y_predict)+1),y_predict ,label='predict')
# plt.plot(range(1,len(y_predict)+1),y_test,label='ture')
# for a, b in zip(range(1,len(y_predict)+1), y_predict):
#     plt.text(a, b-0.5, b, ha='center', va='bottom')
# for a, b in zip(range(1,len(y_predict)+1), y_test):
#     plt.text(a, b+0.2, b, ha='center', va='bottom')
# plt.title("Decision tree regression")
# plt.xlabel("Molecule")
# plt.ylabel("EQE Compare")
# plt.legend()
# plt.show()
# """