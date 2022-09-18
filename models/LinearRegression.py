import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features+1, 1))

    def numerical_solution(self, x, y, epochs, batch_size, lr, optim, batch_gradient=False):

        """
        The numerical solution of Linear Regression
        Train the model for 'epochs' times with minibatch size of 'batch_size' using gradient descent.
        (TIP : if the dataset size is 10, and the minibatch size is set to 3, corresponding minibatch size should be 3, 3, 3, 1)

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            optim : optimizer. (fixed to 'stochastic gradient descent' for this assignment.)

        [Output]
            None

        """
        x_new = self.add_bias(x)

        self.W = self.W.reshape(-1)
        #print('W: ', self.W)
        num_data = len(x)
        num_batch = int(np.ceil(num_data / batch_size))
        #beta = np.array([0,0]) 
        #cost_history = [0]*epochs
        for epoch in range(epochs):
            if batch_gradient:
                # batch gradient descent

                # ========================= EDIT HERE ========================
                #w_update = np.zeros_like(self.W)
                y_pred = np.dot(x_new, (self.W).T)
                loss_vector = y-y_pred
                #hypothesis = x_new(beta)
                #loss_vector = hypothesis - y
                grad = x_new.T.dot(loss_vector)/num_data
                w_update = -(2/num_data)*lr*(np.dot(x_new.T, loss_vector))
                self.W = w_update
                #cost = np.sum((x_new.dot(beta)-y)**2)/2/num_data
                #cost_history[epoch] = cost
                #grad = None


                # ============================================================

                self.W = optim.update(self.W, grad, lr)
            else:
                # mini-batch stochastic gradient descent
                for batch_index in range(num_batch):
                    batch_x = x_new[batch_index*batch_size:(batch_index+1)*batch_size]
                    batch_y = y[batch_index*batch_size:(batch_index+1)*batch_size]

                    num_samples_in_batch = len(batch_x)
                    
                    # ========================= EDIT HERE ========================
                    random_index = np.random.randint((batch_index+1)*batch_size - batch_index*batch_size)
                    sampX = batch_x[random_index:random_index+1]
                    sampY = batch_y[random_index:random_index+1]
                    loss_vector = sampX.dot(self.W)-sampY
                    grad = 2*sampX.T.dot(loss_vector)
                    # ============================================================
                    # cited: https://better-tomorrow.tistory.com/entry/Stochastic-gradient-descent%ED%99%95%EB%A5%A0%EC%A0%81-%EA%B2%BD%EC%82%AC-%ED%95%98%EA%B0%95%EB%B2%95

                    self.W = optim.update(self.W, grad, lr)

    def analytic_solution(self, x, y):
        """
        The analytic solution of Linear Regression
        Train the model using the analytic solution.

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )

        [Output]
            None

        [Hints]
            1. Use np.transpose for transposing a matrix.
            2. Use np.linalg.inv for making a inverse matrix.
            3. Use np.dot for performing a dot product between two matrices.
        """
        x_new = self.add_bias(x)
        #y = 2 + 3*x
        # ========================= EDIT HERE ========================
        #y = y.reshape(x_new.shape[0], 1)
        beta = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y)
        

        self.W = beta
        # ============================================================
    
    def add_bias(self, x):
        # ========================= EDIT HERE ========================
        # You should add column of ones for bias after the last column of x
        _, columns = x.shape
        bias = np.ones(_)
        print(len(bias))
        x_new = np.c_[x, bias]
        # ========================= EDIT HERE ========================
        return x_new

    def eval(self, x):
        """
        Evaluation Function
        [Input]
            x : input for linear regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )
        """
        x_new = self.add_bias(x)

        pred = np.dot(x_new, self.W)

        return pred
