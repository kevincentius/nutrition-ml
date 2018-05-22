
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection._validation import cross_validate, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

import numpy as np
from sklearn.linear_model.ridge import Ridge

import matplotlib.pyplot as plt

def read_csv_data(data_path):
    data = np.loadtxt(data_path)
    
    num_features = len(data[0]) - 1
    
    x = data[:, 0:num_features]
    y = data[:, num_features]
    
    return x, y


if __name__ == '__main__':
    nela = 'D:/master project/data/newsela/data.csv'
    cepp = 'D:/master project/data/CEPP/data.csv'
    
    regr = linear_model.LinearRegression(normalize=True)
    #regr = linear_model.Ridge(alpha=0.001, normalize=True)
    
    # train on first corpus
    x_train, y_train = read_csv_data(cepp)
    regr.fit(x_train, y_train)
    
    # test on second corpus
    x_test, y_test = read_csv_data(nela)
    predict = regr.predict(x_test)
    
    plt.scatter(y_test, predict)
    plt.show()
    