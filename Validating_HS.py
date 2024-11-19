import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from imodels import HSTreeRegressorCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


HS = HSTreeRegressorCV()
DT = DecisionTreeRegressor(random_state=99)

diabetes = datasets.load_diabetes()
X = diabetes.data
Y = diabetes.target

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=1/3,random_state=99)

leafs = np.array([2,4,8,12,15,20,24,28,30,32])
r2_scores_dt = []
r2_scores_hs = []
mse_dt = []
mse_hs = []



for num_leaves in leafs:
    DT.set_params(max_leaf_nodes=num_leaves)
    DT.fit(train_x, train_y)
    predictions = DT.predict(test_x)
    r2 = r2_score(test_y, predictions)
    mse1 = mean_squared_error(test_y, predictions)
    r2_scores_dt.append(r2)
    mse_dt.append(mse1)

for num_leaves in leafs:
    HS = HSTreeRegressorCV(max_leaf_nodes=num_leaves)
    HS.fit(train_x, train_y)
    predictions = HS.predict(test_x)
    r2_hs = r2_score(test_y, predictions)
    mse2 = mean_squared_error(test_y, predictions)
    r2_scores_hs.append(r2_hs)
    mse_hs.append(mse2)


######################## R2 SCORE ########################
plt.figure(figsize=(10,6))
plt.plot(leafs, r2_scores_dt, marker='o', linestyle='-', color='b', label="DT R2 Score")
plt.plot(leafs, r2_scores_hs, marker='o', linestyle='-', color='red', label='HS R2 Score')
plt.xlabel("Number of Leaves")
plt.ylabel("R2 Score")
plt.grid(True)
plt.title("R2 Score")
plt.legend()
plt.show()

######################## MSE ########################
plt.figure(figsize=(10, 6))
plt.plot(leafs, mse_dt, marker='o', linestyle='-', color='b', label='DT MSE')
plt.plot(leafs, mse_hs, marker='o', linestyle='-', color='red', label='HS MSE')
plt.xlabel('Number of Leaves')
plt.ylabel('MSE')
plt.grid(True)
plt.title("Mean Squared Error")
plt.legend()
plt.show()

