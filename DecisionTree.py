from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load


data = load_iris()     #['data]

X = data['data']
y = data['target']

#%%

train_X, test_X, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=44)

#%%

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_X,train_y)

#%%

pred_y = classifier.predict(test_X)
accuracy = accuracy_score(test_y, pred_y)

print(accuracy)
pred_y_train = (classifier.predict(train_X))

train_accuracy = accuracy_score(train_y,pred_y_train)
print(train_accuracy)

print(data['target_names'][pred_y])

#%%

dump(classifier, 'IrisClassifier.joblib') 
