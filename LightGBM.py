import pandas as pd
import numpy as np
import lightgbm as lgb
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

MSEs = []
MAEs = []
R2s = []
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    d_train = lgb.Dataset(x_train, label=y_train)
    params = {}
    params['learning_rate'] = 0.003
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['sub_feature'] = 0.5
    params['num_leaves'] = 10
    params['min_data'] = 50
    params['max_depth'] = 10
    clf = lgb.train(params, d_train, 100)



    # 预测
    y_predict = clf.predict(x_test)

    MSE = mean_squared_error(y_test, y_predict)
    MAE = mean_absolute_error(y_test, y_predict)
    R2 = r2_score(y_test, y_predict)
    MSEs.append(MSE)
    MAEs.append(MAE)
    R2s.append(R2)
print(np.mean(MSEs))
print(np.mean(MAEs))
print(np.mean(R2s))

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