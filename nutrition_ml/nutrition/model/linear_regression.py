
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection._validation import cross_val_score

import matplotlib.pyplot as plt
from nutrition.structure.data_set import DataSet


if __name__ == '__main__':
    data_set = DataSet('cepp')
    
    x = data_set.load_feature_matrix('2018-05-22_8-features')
    y = data_set.data['labels']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True, random_state=0)
    
    regr = linear_model.LinearRegression(normalize=True)
    #regr = linear_model.Ridge(alpha=0.001, normalize=True)
    
    regr.fit(x_train, y_train)
    predict = regr.predict(x_test)
    
    scores = cross_val_score(regr, x_test, y_test, scoring='neg_mean_squared_error')
    print(scores.mean())
    
    plt.scatter(y_test, predict)
    plt.show()
    