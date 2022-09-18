import numpy as np
from models.LinearRegression import LinearRegression

import matplotlib.pyplot as plt
from utils import optimizer, RMSE, load_data


np.random.seed(2021)

# Data generation
train_data, test_data = load_data('Wave')
x_train_data, y_train_data = train_data[0], train_data[1]
x_test_data, y_test_data = test_data[0], test_data[1]

# Hyper-parameter
_epoch=1000
_optim = 'SGD'

# ========================= EDIT HERE ========================
"""
Choose param to search. (batch_size or lr)
Specify values of the parameter to search,
and fix the other.
e.g.)
search_param = 'lr'
_batch_size = 32
_lr = [0.1, 0.01, 0.05]
"""
search_param = 'batch_size'
_batch_size = 32 if search_param == 'lr' else [4,8,16,32,64]
_lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001] if search_param == 'lr' else 0.01


#search_param = 'batch'
#_lr = 0.01
#_batch_size = [4, 8, 16, 32, 64]
# ============================================================


train_results = []
test_results = []
search_space = _lr if search_param == 'lr' else _batch_size
#print(search_space)
for i, space in enumerate(search_space):
    # Build model
    model = LinearRegression(num_features=x_train_data.shape[1])
    optim = optimizer(_optim)

    # Train model with gradient descent
    if search_param == 'lr':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=space, optim=optim)
    else:
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=space, lr=_lr, optim=optim)
    
    ################### Evaluate on train data
    # Inference
    inference = model.eval(x_train_data)

    # Assess model
    error = RMSE(inference, y_train_data)
    print('[Search %d] RMSE on Train Data : %.4f' % (i+1, error))

    train_results.append(error)

    ################### Evaluate on test data
    # Inference
    inference = model.eval(x_test_data)

    # Assess model
    error = RMSE(inference, y_test_data)
    print('[Search %d] RMSE on Test Data : %.4f' % (i+1, error))

    test_results.append(error)

"""
Draw scatter plot of search results.
- X-axis: search paramter
- Y-axis: RMSE (Train, Test respectively)

Put title, X-axis name, Y-axis name in your plot.

Resources
------------
Official document: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html
"Data Visualization in Python": https://medium.com/python-pandemonium/data-visualization-in-python-scatter-plots-in-matplotlib-da90ac4c99f9
"""

num_space = len(search_space)
plt.scatter(search_space, train_results, label='train', marker='x', s=150)
plt.scatter(search_space, test_results, label='test', marker='o', s=150)
plt.legend()
plt.title('Search results')
plt.xlabel(search_param)
plt.ylabel('RMSE')
plt.show()
