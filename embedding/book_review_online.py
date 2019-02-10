# sample code to use glove to predict book review
# downlaod the Pre-trained word vectors and unzip it
# wget http://nlp.stanford.edu/data/glove.6B.zip

import os
import numpy as np
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

# doc/words and its label about books
docs = ['The book is an easy read. The code all works',                                                                 'This is well written book about Docker. It contains of4 books so chances to get more ideas and tips. I enjoy reading this book.',
    'Very useful, well done Docker book. Good example',
    'The book does an incredible job of slowly building on previous examples to help you get comfortable with the core concepts of Docker. ',
    'A great book for learning Docker and getting your feet wet',
    'There is no code or examples, or even diagrams. All of the discussion is in the abstract. I did not find this book helpful',
    'Lack of any useful details',
    'A pathetic, rip-off of a book',
    'Not a beginner book. Poorly written',
    'It sounds like a trivial task, but all the examples on git hub and docker registry all did not work']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])

max_words = 10000 # consider only top 10K words from our data

t = Tokenizer( num_words=max_words )
t.fit_on_texts(docs)
word_to_index = t.word_index
print( word_to_index )
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
vocab_size = len(word_to_index) + 1

max_length = 30 # max length of our sentences
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# glove has 50,100,200 dimesnions word embedding
# here we use 50 dimension's word embedding
glove_dir = '../data/glove.6B'
# load content into this RAM cache
embeddings_index = {}
#
f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# load our example data using GloVe's 50d word embedding
embedding_dim = 50
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_to_index.items():
  if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      # That means we set word ( as integer i) using glove's word embedding as embedding_vector
      embedding_matrix[i] = embedding_vector

print(embedding_matrix)

model = Sequential()
# use pre-train word embedding for the word in our sentence
model.add(Embedding(max_words, embedding_dim, input_length=max_length))
model.add(Flatten())
# our later layers is mainly focus sentence meaning
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# really load the pre-train data, and make it non-trainable
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

#model.load_weights(self.weights_filename);
history = model.fit(padded_docs, labels, epochs=50, verbose=0)
model.save_weights('pre_trained_glove_model.h5')

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# just to test our model
test_docs = ['The books is great', 'it is so badly written'] 
test_encoded_docs = t.texts_to_sequences(test_docs)
test_padded_docs = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')
print( t.word_index )
print( test_encoded_docs )
print( test_padded_docs )

test_results = model.predict( test_padded_docs )
print( "test result is " );
print( test_results );

# Or access the embedding layer through the constructed model
# first `0` refers to the position of embedding layer in the `model`
embeddings = model.layers[0].get_weights()[0]

# `embeddings` has a shape of (num_vocab, embedding_dim)

# try to plot it
labels = []
data_x = []
data_y = []

# plot all the words we learned
for k,v in word_to_index.items():
#  print( k  )
  labels.append(k)
#  print( v  )
  # embedding need a list as input, thus need to pass [v] here
  # so embeddings[ [v] ] will output like this: [ [0.1 0.2] ]
  tmp = embeddings[ [v] ][0]
#  print( tmp )
  x = tmp[0]
  y = tmp[1]
#  print( x )
#  print( y )
  data_x.append( x )
  data_y.append( y )

plt.plot(data_x, data_y, 'ro')

# add label to each word point in the (x,y) space
for label, x, y in zip(labels, data_x, data_y):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()

