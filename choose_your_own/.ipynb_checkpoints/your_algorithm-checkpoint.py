#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

#number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=80, num=10)]
#number of features to consider everytime there is a split
max_features = ['auto', 'sqrt']
#maximum number of levels in a tree
max_depth = [2, 4]
#minimum number of samples required to split a node
min_samples_split = [2, 5]
#minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
#method of selecting samples for training each tree
bootstrap = [True, False]
param_grid = {
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'bootstrap':bootstrap
}

#create random forest classifier
rfc = RandomForestClassifier()

#create gridsearchcv classifier
clf = GridSearchCV(rfc, param_grid)

clf.fit(features_train, labels_train)

print(clf.score(features_test, labels_test))





try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
