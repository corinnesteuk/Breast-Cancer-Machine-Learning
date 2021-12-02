from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from random import randrange
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score



train = pd.read_csv('BreastCancerTrain.csv')
test = pd.read_csv('BreastCancerTest.csv')
train_y = train['Label']
train_X = train.drop(['Label', 'Unnamed: 0'], axis = 1)
train_X, dev_X, train_y, dev_y = train_test_split(train_X, train_y, test_size=0.15, random_state=123)

test_y = test['Label']
test_X = test.drop(['Label', 'Unnamed: 0'], axis = 1)


'''Logistic Regression'''
def logistic(X, y, C = 1, penalty = 'l2'):
#Standardizing the training and testing data   
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)


    lr = LogisticRegression(C = C, penalty = penalty, solver='liblinear')
    lr.fit(X_std, y)

    prob_arrays = lr.predict_proba(X_std[:,])
    max_prob_indices = lr.predict_proba(X_std[:,]).argmax(axis = 1)
    class_pred = lr.predict(X_std[:, :])

    return class_pred

def printMetrics(actual, predictions):
    '''
    Description:
        This method calculates the accuracy of predictions
    '''
    assert len(actual) == len(predictions)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return (correct / float(len(actual)) * 100.0)


penalty = ['l1', 'l2'] 
C = [0.001,.009,0.01,.09,1, 5, 10, 15]

def find_best_model(X, y, C_list, penalty_list):
    acc_list = []
    for p in penalty:
        for c in C:        
            pred = logistic(X, y , C = c, penalty = p)
            acc_list.append([p, c, printMetrics(y.values, pred)])
    return acc_list

 
'''Training Set'''   
#First, we are finding the best model
a = find_best_model(train_X, train_y, C, penalty)
df = pd.DataFrame(a, columns = ['Penalty', "C Values", "Accuracy"])
m = max( df['Accuracy'].values)
best_model = df.where(df["Accuracy"] == m)
best_model = best_model.dropna()
print(best_model)

lr = LogisticRegression(C = 15, penalty = 'l1', solver='liblinear')
lr.fit(train_X, train_y)
tr = lr.predict(train_X)
train_y = train_y.reset_index()
train_acc = printMetrics(train_y['Label'], tr)
print(train_acc)

'''Development Set'''
#Finding best model of the development set (same as the training)
b = find_best_model(dev_X, dev_y, C, penalty)
df_dev = pd.DataFrame(b, columns = ['Penalty', "C Values", "Accuracy"])
m = max( df_dev['Accuracy'].values)
best_model_dev = df.where(df["Accuracy"] == m)
best_model_dev = best_model.dropna()
print(best_model_dev)

dev = lr.predict(dev_X)
dev_y = dev_y.reset_index()
dev_acc = printMetrics(dev_y['Label'], dev)
print(dev_acc)

'''Test Set'''
prob_arrays = lr.predict_proba(test_X)
max_prob_indices = lr.predict_proba(test_X).argmax(axis = 1)
test = lr.predict(test_X)
test_acc = printMetrics(test_y, test)
print(test_acc)

ar = [train_acc, dev_acc, test_acc]

output = pd.DataFrame(ar, index = ["Train", "Development", "Test"], columns = ['Accuracy'])
print(output)

output.to_csv("Logistic Regression Accuracy Output")

# Final test accuracy: .9824561403508771
