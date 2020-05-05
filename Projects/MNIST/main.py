from sklearn.datasets import fetch_openml
import winsound

def makeSound():
    winsound.MessageBeep(winsound.MB_OK)

# MNIST - multiclass classification nexample
# Get the data from source
mnist = fetch_openml('mnist_784', version=1)

# Get the data and target
X, y = mnist["data"], mnist["target"]

# Split the train and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]



## Model selection
from sklearn.model_selection import cross_val_score

print ("Training SVM OvO")
# 1. SVM classifier :: OvO strategy(default)
from sklearn.svm import SVC
svm_clf = SVC()
svm_ovo_score = cross_val_score(svm_clf, X_train, y_train, cv=3, scoring="accuracy")
print("svm_ovo_score", svm_ovo_score) # svm_ovo_score [0.977  0.9738 0.9739]
makeSound()


print ("Training SVM OvR")
# 2. SVM classifier :: OvR strategy
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
svm_ovr_score = cross_val_score(ovr_clf, X_train, y_train, cv=3, scoring="accuracy")
print("svm_ovr_score: ", svm_ovr_score ) # svm_ovr_score:  [0.97685 0.9738  0.97495]
makeSound()


print ("Training SGDClassifier")
# 3. SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print("sgd_score: ", sgd_score) #sgd_score:  [0.87365 0.85835 0.8689 ]
makeSound()


print ("Training RandomForestClassifier")
# 4. RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_score = cross_val_score(rf_clf, X_train, y_train, cv=3, scoring="accuracy")
print("rf_score", rf_score) # rf_score [0.9646  0.96255 0.9666 ]
makeSound()


print ("Training KNeighborsClassifier")
# 5. KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_score = cross_val_score(kn_clf, X_train, y_train, cv=3, scoring="accuracy")
print("kn_score", kn_score) # kn_score [0.9676  0.9671  0.96755]
makeSound()

# Grid search for KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
#
# param_grid= [
#     {'weights': ['uniform', 'distance'],
#      'n_neighbors': list(range(1,10, 2))}
# ]
#
# grid_search = GridSearchCV(kn_clf, param_grid=param_grid, cv=3)
# grid_search.fit(X_train, y_train)
# print("best_params_", grid_search.best_params_)


## OMG, gonna take a long long time
### Make some sound once this all finish
makeSound()
