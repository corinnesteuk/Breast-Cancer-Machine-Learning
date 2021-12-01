
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# read in the test set
test = pd.read_csv('BreastCancerTest.csv')

y_test = test['Label']
X_test = test.drop('Label', axis = 1)

# check counts of each class in the test set
print(y_test.value_counts())


# compute accuracy of predicting every mass as benign 
bl_pred = np.repeat(np.array(["B"]), [114], axis=0)
bl_pred = pd.Series(bl_pred)

print(accuracy_score(y_test, bl_pred))

# accuracy: 0.6403508771929824