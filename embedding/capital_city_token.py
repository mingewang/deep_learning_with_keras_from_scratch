# keras sample to word embedding using Tokenizer
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

# doc/words and its label
docs = ['China',
    'Italy',
    'Germany',
    'USA',
    'Canada',
    'Beijing',
    'Rome',
    'Berlin',
    'Washington DC',
    'Ottawa']

# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])


# we use integer to encode/represent the documents's word
# here we use Keras text preprocessing Tokenizer
t = Tokenizer()
# fit_on_texts() will updates t's internal vocabulary based on a list of texts
t.fit_on_texts(docs)
# t.word_index will looks like this
#{'US': 8, 'China': 7, ... ]
word_to_index = t.word_index
print("word_to_index is:")
print(word_to_index)


# texts_to_sequences will transforms each text in texts in a sequence of integers.
encoded_docs = t.texts_to_sequences(docs)
# encoded_docs will look this 
#[[6, 2], [3, 1], [7, 4], [8, 1], [9], [10], [5, 4], [11, 3], [5, 1], [12, 13, 2, 14]]

# Another useful function is called one_hot(): 
# It encodes a text into a list of word indexes of size n.
# in other word it creates a hash of each word 
# as an efficient integer encoding.
# for example to achieve the similar things, we can do this:

# vocab_size = 50
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
# then we get each word from our defined document
# using list comprehension ( list of list )
#   words = [ item for d in docs for item in d.split()]
# get word's integer encoding/representation
# word_to_index = { d:one_hot(d, vocab_size) for d in words}

# now at this time we should appreciate what Tokenizer give us


print("encoded_docs is:")
print(encoded_docs)

# Each doc/sequences could have different lengths 
# and Keras prefers inputs to be vectorized 
# and all inputs to have the same length. 
# here we will pad all input sequences to have the length of 2 
# pad documents to a max length of 2 words
max_length = 2 
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print("padded_docs is:")
print(padded_docs)

# word_to_index start from 1,
# while we use will use 0 as the padding value,
# so our vocabulary size need to at lease add this one
# in order to feed into embedding layer
vocab_size = len(word_to_index) + 1

# define the model
model = Sequential()
# embedding layers 
# first parameter: size of the vocabulary, i.e. maximum integer index + 1.
# second parameter: output_dim, Dimension of the dense embedding. 
# We will choose a small embedding space of 2 dimensions for easy plotting
model.add(Embedding(vocab_size, 2, input_length=max_length))
# it will output [[x1,y1],[x2,y2],... ]
# flatten just make it easy to Dense layer to classifiy
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# we will try to see embedding's output directly by access embedding layer
# if you have access to the embedding layer explicitly
#embeddings = model.get_layer("embedding_1").get_weights()[0]

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
  #print( k  )
  labels.append(k)
  #print( v  )
  print("city is:", k, ", index inside embedding layer: ", v)
  # embedding need a list as input, thus need to pass [v] here
  # so embeddings[ [v] ] will output like this: [ [0.1 0.2] ]
  tmp = embeddings[ [v] ][0]
  print("position in embedded space:",  tmp, "\n" )
  x = tmp[0]
  y = tmp[1]
  #print( x )
  #print( y )
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
