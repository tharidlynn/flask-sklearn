from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
iris = load_iris()

X_iris = iris.data
y_iris = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris)

clf = LogisticRegression()
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.pkl')