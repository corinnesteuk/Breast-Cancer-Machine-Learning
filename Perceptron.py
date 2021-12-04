#SET UP ENVIRONMENT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



#LOAD DATASET
train_set = pd.read_csv('BreastCancerTrain.csv')
test_set = pd.read_csv('BreastCancerTest.csv')



#PREPARE AND SPLIT DATASETS
#initial train_sets
X = train_set.drop(['Label', 'Unnamed: 0'], axis = 1)
y = train_set['Label']

#train set and development set
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.15, random_state=1)

#test set
X_test = test_set.drop(['Label', 'Unnamed: 0'], axis = 1)
y_test = test_set['Label']



#STANDARDIZE DATA
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_dev_std = sc.transform(X_dev)
X_test_std = sc.transform(X_test)



#TRAIN PERCEPTRON - train classifier with default params on training set, evaluate using development set
ppn = Perceptron()
ppn.fit(X_train_std, y_train)
y_dev_pred = ppn.predict(X_dev_std)
print('Default Perceptron Accuracy: %.3f' % accuracy_score(y_dev, y_dev_pred))  #accuracy: 0.971 %



#TUNE PERCEPTRON HYPERPARAMETERS - tune hyperparameters using GridSearchCV
#note: lower values of epochs (max_iter) during Grid Search will result in warning: "maximum number of iteration reached before convergence."
#meaning that perceptron did not converge for the max_iter param value (max_iter <100) and therefore # of epochs or max_iter must be increased, however for our purposes we can ignore warning.
import warnings
warnings.filterwarnings("ignore")

eta0_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]         #learning rate parameter
penalty_values = ['l1', 'l2', 'None']                                     #regularization parameter
max_iter_range = [1, 10, 100, 1000, 10000]                                #maximum iteration (epochs) parameter

#set up grid and perform grid search
grid = [{'eta0': eta0_range, 'penalty' : penalty_values, 'max_iter': max_iter_range}]
grid_search = GridSearchCV(Perceptron(random_state=1), grid, scoring='accuracy')
grid_search = grid_search.fit(X_train_std, y_train)

print('Grid Search Best Parameters: ' + str(grid_search.best_params_) )    #best parameters: {'eta0': 0.01, 'max_iter': 100, 'penalty': 'l1'}
print('Grid Search Best Score %.3f' % grid_search.best_score_)             #best score: 0.977 %

#tuned perceptron
tuned_ppn = grid_search.best_estimator_

grid_dev_pred = tuned_ppn.predict(X_dev_std)
print('Tuned Perceptron Accuracy: %.3f' % accuracy_score(y_dev, grid_dev_pred))  #accuracy: 0.971 %



#TEST PERCEPTRON - retrain classifier on combined train+dev set using tuned hyperparameters and evaluate performance on test set
X_std = sc.transform(X)                                                          #standardize combined train+dev set

final_ppn = Perceptron(eta0=0.01, max_iter=100, penalty="l1")
final_ppn.fit(X_std, y)

y_test_pred = final_ppn.predict(X_test_std)
print('Final Perceptron Accuracy: %.3f' % accuracy_score(y_test, y_test_pred))   #accuracy: 0.991 %



#ACCURACY & GRID SEARCH RESULTS - Overall accuracy scores and listed accuracy scores for each combination of parameters from grid search - result details found in Perceptron Results file
means = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
with open("Perceptron_Results.txt", "w") as results:

    results.write('MODEL ACCURACY RESULTS:' + '\n')
    results.write('Default Perceptron Accuracy: %.3f' % accuracy_score(y_dev, y_dev_pred) + '\n')
    results.write('Grid Search Best Parameters: ' + str(grid_search.best_params_) + '\n')
    results.write('Grid Search Best Score %.3f' % grid_search.best_score_ + '\n')
    results.write('Tuned Perceptron Accuracy: %.3f' % accuracy_score(y_dev, grid_dev_pred) + '\n')
    results.write('Final Perceptron Accuracy: %.3f' % accuracy_score(y_test, y_test_pred) + '\n')

    results.write('\n')

    results.write('GRID SEARCH RESULTS:' + '\n')
    for mean, param in zip(means, params):
        results.write(">%.3f with: %r" % (mean, param) + '\n')

results.close()








