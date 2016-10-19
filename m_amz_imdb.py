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

#import libpos
from libspacy import get_adjectives
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers import Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D 
from keras.callbacks import  ModelCheckpoint 

random.seed(0)
#numpy.set_printoptions(threshold=numpy.inf)
train_data=[]
cx=0
num_train_samples=1200
num_test_samples=400

print "Loading training data from amazon"
#Total train plus test should be 1600
num_samples_to_drop = 100
num_samples = num_train_samples + num_test_samples 
with open("datasets/amazon/deceptive.csv") as csvfile:
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

#raw_test_data = train_data[-num_test_samples:]
#raw_train_data = train_data[:num_train_samples]

raw_train_data = train_data

print len(raw_train_data)


#Split the train into training and labels and preprocess the review
train_data=[]
train_labels=[]
for (review, label) in raw_train_data:
  train_data.append(review_to_word(review))
  train_labels.append(int(label))
amz_train_data = train_data
amz_train_labels = train_labels
#print train_data[0], train_labels[0]
#print test_data[0], test_labels[0]
#Begin loading amazon hotel reviews
print "Loading testing data from imdb"
#Total train plus test should be 1600
imdb_num_test_samples=3000
imdb_test_data = []
with open("datasets/imdb/labeledTrainData.tsv") as csvfile:
  reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
  for row in reader:
    if len(imdb_test_data) >= imdb_num_test_samples:
      break
    imdb_test_data.append(row)

print len(imdb_test_data)
#Now shuffle the data

random.shuffle(imdb_test_data)

raw_test_data = imdb_test_data

print  len(raw_test_data)



test_data=[]
test_labels=[]
for (_, label, review) in raw_test_data:
  test_data.append(str(review_to_word(review)))
  test_labels.append(int(label))
imdb_test_data = test_data
imdb_test_labels = test_labels
#End of amazon hotel reviews

print "Size of training amazon_deceptive=", len(amz_train_data), len(amz_train_labels)
print "Size of testing imdb=", len(imdb_test_data), len(imdb_test_labels)

#Combine the training and test to give it to autoencoder

combined_train_data = amz_train_data + imdb_test_data
print "Combined data=", len(combined_train_data)
#Find the adjectives in the training to construct a feature list
print "Extracting features"
adjectives=set()

for review in combined_train_data:
  adjs = get_adjectives(review)
  adjectives.update(set(adjs)) #get_adjectives returns a set

#print sorted(adjectives)
print "Number of features=", len(adjectives)

#Generate X_train and X_test from these features
X_train=[]
for review in combined_train_data:
  sample = [0 for i in adjectives]
  for (i,adj) in enumerate(adjectives):
    sample[i] = review.count(adj)
  X_train.append(sample)

X_test=[]
for review in imdb_test_data:
  sample = [0 for i in adjectives]
  for (i,adj) in enumerate(adjectives):
    sample[i] = review.count(adj)
  X_test.append(sample)


print len(X_train), len(X_train[0])
print len(X_test), len(X_test[0])

input_dim = len(adjectives)
encoding_dim = int(input_dim/2)

#X_train = numpy.array(X_train)
#X_test = numpy.array(X_test)

input_img = Input(shape=(input_dim,)) #placeholder
encoded = Dense(encoding_dim, activation='relu',init='he_uniform')(input_img) #Gaussian initialization scaled by fan_in (He et al., 2014)

decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
  metrics=['accuracy' ])

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

#---------------------------------
# Train the autoencoder
#---------------------------------

checkpointer = ModelCheckpoint(filepath=os.path.join('.',r'./std_model.h5'), verbose=0, save_best_only=True) 

out1 = autoencoder.fit(X_train, X_train,
                nb_epoch=50, #This should take about 3 minutes
                batch_size=64,
                shuffle=True,
                verbose=False,
                validation_data=(X_test, X_test),
                callbacks=[checkpointer])

X_test_encoded = encoder.predict(X_test)
print len(X_test_encoded)
print X_test_encoded.shape
