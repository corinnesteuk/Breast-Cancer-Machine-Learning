
#imports
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

#dataset
dataset = pd.read_csv('BreastCancer.csv')

#test train split
train = pd.read_csv('BreastCancerTrain.csv')
y_train = train['Label']
X_train = train.drop('Label', axis = 1)
test = pd.read_csv('BreastCancerTest.csv')
y_test = test['Label']
X_test = test.drop('Label', axis = 1)
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 20)

#SVM 
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

#performance metrics
print("regular SVM performance metrics:")
print(classification_report(y_test, y_predict))

#lets try to improve this!

#normalization
X_train_min = X_train.min()
X_train_min
X_train_max = X_train.max()
X_train_max
X_train_range = (X_train_max- X_train_min)
X_train_range
X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()
X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

#SVM on normalized data
svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

#performace metrics on normalized data
print("normalized data SVM performance metrics:")
print(classification_report(y_test,y_predict))

#lets try to improve this!

#grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

#best parameters and estimators
print (grid.best_params_)
print ('\n')
print (grid.best_estimator_)

#performace metrics on grid search parameters
grid_predictions = grid.predict(X_test_scaled)
print("grid search SVM performance metrics:")
print(classification_report(y_test,grid_predictions))

#regular SVM accuracy: 0.95
#normalized data SVM accuracy: 0.97
#grid search SVM accuracy: 0.97
        
