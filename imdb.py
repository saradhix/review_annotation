import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "functions/"))
import numpy
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from clean_data import *
from sklearn.semi_supervised import label_propagation
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import random

import libpos
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
random.seed(0)
#numpy.set_printoptions(threshold=numpy.inf)
train_data=[]
cx=0
num_train_samples=5000
num_test_samples=1000

print "Loading data"
#Total train plus test should be 1600
num_samples_to_drop = 100
num_samples = num_train_samples + num_test_samples 
with open("datasets/imdb/labeledTrainData.tsv") as csvfile:
  reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
  for row in reader:
    if len(train_data) >= num_samples:
      break
    train_data.append(row)

print len(train_data)
#Now shuffle the data

#print train_data[0]
random.shuffle(train_data)
#print train_data[0]

#Split the train into train and test

raw_test_data = train_data[-num_test_samples:]
raw_train_data = train_data[:num_train_samples]

print len(raw_train_data), len(raw_test_data)


#Split the train into training and labels and preprocess the review
train_data=[]
train_labels=[]
for (_, label, review) in raw_train_data:
  train_data.append(review_to_word(review))
  train_labels.append(int(label))

print "Print train_data sample"
print train_data[0], train_labels[0]

test_data=[]
test_labels=[]
for (_, label, review) in raw_test_data:
  test_data.append(str(review_to_word(review)))
  test_labels.append(int(label))

print "Print test sample"
print test_data[0], test_labels[0]

#Find the adjectives in the training to construct a feature list
print "Extracting features"
adjectives=set()

for review in train_data:
  adjs = libpos.get_adjectives(review)
  adjectives.update(set(adjs)) #get_adjectives returns a set

#print sorted(adjectives)
print "Number of features=", len(adjectives)

#Generate X_train and X_test from these features
X_train=[]
for review in train_data:
  sample = [0 for i in adjectives]
  for (i,adj) in enumerate(adjectives):
    sample[i] = review.count(adj)
  X_train.append(sample)

X_test=[]
for review in test_data:
  sample = [0 for i in adjectives]
  for (i,adj) in enumerate(adjectives):
    sample[i] = review.count(adj)
  X_test.append(sample)


print len(X_train), len(X_train[0])
print len(X_test), len(X_test[0])


#Now run a machine learning algorithm for train and test
print "Starting training"
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, train_labels) 
print "Making predictions..."
predicted_labels = clf.predict(X_test)
cm = confusion_matrix(test_labels, predicted_labels)
print cm
print(classification_report(test_labels, predicted_labels))

#Now run a neural network
print "Number of features=",len(adjectives)
model = Sequential()
model.add(Dense(120, input_dim=len(adjectives), init='uniform', activation='relu'))
model.add(Dense(40, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, train_labels, nb_epoch=15, batch_size=10)
# evaluate the model
scores = model.evaluate(X_train, train_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print "Running predictions for support"
predicted_labels = model.predict(X_test)
predicted_labels = [int(round(x)) for x in predicted_labels]
cm = confusion_matrix(test_labels, predicted_labels)
print cm
print(classification_report(test_labels, predicted_labels))
