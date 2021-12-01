
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# split the data into train and test sets
train = pd.read_csv('BreastCancerTrain.csv')
print(train.head())
print(train.shape)

y_train = train['Label']
X_train = train.drop('Label', axis = 1)

test = pd.read_csv('BreastCancerTest.csv')
print(test.head())
print(test.shape)

y_test = test['Label']
X_test = test.drop('Label', axis = 1)



# perform cross validation to select a value for k between 1 and 10
k_vals = range(1, 10)
print(type(k_vals))

k_accuracies = {}

for k in k_vals: 

    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    k_accuracies[k] = scores.mean()

print(k_accuracies)

# find the value for k that returned the lowest average cross validation error (highest accuracy)
max_k = max(k_accuracies, key=k_accuracies.get)
print(max_k)

# train a final KNN classifier using the selected value for k
final_clf = KNeighborsClassifier(n_neighbors = max_k)
final_clf.fit(X_train, y_train)

# report the results on the test set
print(final_clf.predict(X_test))
print(final_clf.score(X_test, y_test))


# final value: k = 7
# accuracy on the test set: 0.9736842105263158
