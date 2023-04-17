from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from joblib import dump
import streamlit as st

# Load IRIS dataset
data = load_iris()     

# Set X and y for training and testing
X = data['data']
y = data['target']

#%%

train_X, test_X, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=44)

#%%

classifier = tree.DecisionTreeClassifier(min_samples_split=6,
                                         min_samples_leaf=3)
                                         
classifier = classifier.fit(train_X,train_y)

#%%

pred_y = classifier.predict(test_X)
accuracy = accuracy_score(test_y, pred_y)
print("\nTest accuracy: %.1f%%" % accuracy)

pred_y_train = (classifier.predict(train_X))
train_accuracy = accuracy_score(train_y,pred_y_train)
print("\Train accuracy: %.1f%%" % train_accuracy)


print(data['target_names'][pred_y])

#%%

dump(classifier, 'IrisClassifier.joblib') 

#%%

# compute the number of labels
num_labels = len(np.unique(train_y))

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# Normalize the data 

train_X = train_X / train_X.max()
test_X = test_X / test_X.max()

# network parameters
batch_size = 4
hidden_units = 64
dropout = 0.45

# model is a 3-layer MLP with ReLU and dropout after each layer
model = Sequential()
model.add(Dense(hidden_units, input_dim=train_X.shape[1]))
model.add(Activation('relu'))
model.add(Dense(hidden_units / 2))
model.add(Activation('relu'))
model.add(Dense(num_labels))

# this is the output for one-hot vector
model.add(Activation('softmax'))
model.summary()

# loss function for one-hot vector
# use of sgd optimizer with default lr=0.01
# accuracy is good metric for classification tasks

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=100, batch_size=batch_size)

# validate the model on test dataset to determine generalization
loss, acc = model.evaluate(test_X,
                           test_y, 
                           batch_size=batch_size,
                           verbose=False)

print("\nTest accuracy: %.1f%%" % (100.0 * acc))

dump(model, 'NeuralNetwork.joblib') 

















