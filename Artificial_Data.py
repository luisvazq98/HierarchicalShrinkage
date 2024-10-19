import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imodels import HSTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


########################## f(x) ##########################
# Defining function 1
def f(x):
    return x**4 - 22*x**2

def f2(x):
    return -x**3



# Generating x values from -5 to 5 and corresponding y values
x_values = np.linspace(-5, 5, 1000)
y_values = f(x_values)

### Use for gaussian noise with mean=0 and std=20 ###
mean = 0
std_dev = 20
noise = np.random.normal(mean, std_dev, len(x_values))
y_noisy = y_values + noise

# Plotting the function without noise
plt.plot(x_values, y_values)
plt.title('f(x) without noise')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
#plt.savefig('f1(noiseless)')
plt.show()

# Plotting the function with noise
plt.plot(x_values, y_noisy)
plt.title('f(x) with Gaussian Noise')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
#plt.savefig('f(x) (noise)')
plt.show()



# Reshaping noisy data for ANN
x_values = x_values.reshape((len(x_values), 1))
y_noisy = y_noisy.reshape((len(y_noisy), 1))

# Splitting noise data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_values, y_noisy, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125) # 0.125 x 0.8 = 0.1



# fit the model
model = HSTreeRegressor()  # initialize a tree model and specify only 4 leaf nodes
model.fit(x_train, y_train)   # fit model
preds = model.predict(x_test) # discrete predictions: shape is (n_test, 1)



RandomForest = RandomForestRegressor()
summary = RandomForest.fit(x_train, y_train)


# Testing set predictions
pred_test = RandomForest.predict(x_test)
pred_test = pred_test.reshape(200, 1)




# plt.scatter(x_test, y_test)
# plt.title("Random Forest Function Approximation")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(True)
# #plt.savefig("f(x) approximation (noise)")
# plt.show()


plt.scatter(x_test, pred_test)
plt.plot(x_values, y_values, color='orange')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.savefig("RF1")
plt.show()


plt.scatter(x_test, preds)
plt.plot(x_values, y_values, color='orange')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.savefig("HS1")
plt.show()