from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def shuffle_data(X, y):

    Data_num = np.arange(X.shape[0])
    np.random.shuffle(Data_num)

    return X[Data_num], y[Data_num]


#train_test_split from scratch
def train_test_split_scratch(X, y, test_size=0.5, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X, y)
    if test_size <1 :
        train_ratio = len(y) - int(len(y) *test_size)
        X_train, X_test = X[:train_ratio], X[train_ratio:]
        y_train, y_test = y[:train_ratio], y[train_ratio:]
        return X_train, X_test, y_train, y_test
    elif test_size in range(1,len(y)):
        X_train, X_test = X[test_size:], X[:test_size]
        y_train, y_test = y[test_size:], y[:test_size]
        return X_train, X_test, y_train, y_test


iris=load_iris()

X= iris.data 
y= iris.target

#split with built-in function
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=40)

print(X_train.shape)
print(X_test.shape)
#split with built-in function
X_train1,X_test1,y_train1,y_test1 = train_test_split_scratch(X,y,test_size=40)
print(X_train1.shape)
print(X_test1.shape)