
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sys
import os

os.environ['KERAS_BACKEND']='theano'


# In[2]:

from keras.engine.topology import Layer
from keras import initializations
from keras import backend as K


# In[3]:

from collections  import defaultdict
import pandas as pd
import numpy as np


# In[4]:

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):


        M = K.tanh(x)
        alpha = K.dot(M,self.W)#.dimshuffle(0,2,1)

        ai = K.exp(alpha)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return K.tanh(weighted_input.sum(axis=1))
        '''
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)
        '''
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


# In[5]:

# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        '''   
        M = K.tanh(x)
        alpha = K.dot(M,self.W)#.dimshuffle(0,2,1)

        ai = K.exp(alpha)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return K.tanh(weighted_input.sum(axis=1))
        '''
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


# # Clean Data

# In[6]:
datafile ='/home/yinqijin/WorkSpace/2.RNA_Structure_Profile/Orig_data/TRAIN_FILE.TXT'
f = open(datafile,'r')
sentenses = []
labels = []

i =0
for line in f:
    if i%4==0:
        sen = line.split('"')[1]
        #print sen
        sen = sen.replace('<e1>','')
        sen = sen.replace('</e1>','')
        sen = sen.replace('<e2>','')
        sen = sen.replace('</e2>','')
        #print sen
        sentenses.append(sen)
    elif i%4==1:
        labels.append(line.splitlines()[0])
    elif i%4==2:
        #This Commment , it's useless,pass
        pass
    else:
        pass
    
    i += 1
    ''' 
    if i>=12:
        print sentenses
        print labels
        break
   '''
    
f.close()


# In[7]:

labels = np.array(labels)
sentenses = np.array(sentenses)


# In[8]:

print len(sentenses),'==',len(labels)
#print labels[:3]
#print sentenses[:3]


# In[9]:

label_name = np.unique(labels)
print  'labes kind(include \'Other\'): ',len(label_name) 
#print np.array(label_name)

label_dict = defaultdict()
for item in label_name:
    label_dict[item] = len(label_dict)

    
func = lambda item:label_dict[item]
labels_ = pd.Series(labels)
labels_ =labels_.apply(func)
#labes = labels_.tolist()
labels = np.array(labels_)
print labels


# ## shuffle

# In[10]:

np.random.seed(1234)
index = range(len(labels))
np.random.shuffle(index)
#print index
#print labels[:5],sentenses[:5]
labels = labels[index]
sentenses = sentenses[index]
#print labels[:5],sentenses[:5]


# In[11]:

LongestSen = max(sentenses,key= lambda x :len(x))
print len(LongestSen)


# # Sequence to Vector

# In[12]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


# In[13]:

MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# ### Read Word Dictionary

# In[14]:

GLOVE_DIR = '/home/yinqijin/WorkSpace/2.RNA_Structure_Profile/Orig_data/glove.6B.100d.txt'
embeddings_index = {}
f = open(GLOVE_DIR,'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))


# ## process the Sentenses

# In[20]:

# Tokenizer.fit_on_sequences Tokenizer.fit_on_texts,  any different??
tokenizer = Tokenizer(nb_words= MAX_NB_WORDS)
tokenizer.fit_on_texts(sentenses)
sequences = tokenizer.texts_to_sequences(sentenses)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[21]:

word_index = tokenizer.word_index
print word_index.keys()[:4]
print len(word_index.keys())


# In[22]:

data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
model_labels = to_categorical(np.asarray(labels))   #one-hot 编码labels


# In[23]:

print data[1],data.shape[0]
print model_labels[1]


# In[24]:

nb_validation_samples = int(VALIDATION_SPLIT*data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = model_labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = model_labels[-nb_validation_samples:]

print('Traing and validation set number of positive and negative reviews')
print y_train.shape[0]
print y_val.shape[0]


# ## Get Sample Word's Vector Values from trained Word2Vec Dictionary

# In[25]:

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



# # Make Network

# In[30]:

from keras.layers import Embedding
from keras.layers import Dense,Input,Activation
from keras.layers import Embedding, LSTM, Bidirectional,GRU,InputLayer
from keras.models import Model,Sequential
from  keras.regularizers import ActivityRegularizer
from keras.layers.core import Dropout

# In[31]:

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True,
                           dropout=0.3)


# In[32]:



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100, return_sequences=True),merge_mode='sum')(embedded_sequences)
l_att = AttLayer()(l_lstm)
preds = Dense(len( model_labels[0]), activation='softmax',activity_regularizer= ActivityRegularizer(l2=0.005))(l_att)


object_function = ['mean_squared_error',
                   'categorical_crossentropy',
'mean_absolute_error',
'mean_absolute_percentage_error',
'mean_squared_logarithmic_error',
'squared_hinge',
'hinge',
'binary_crossentropy',
'kullback_leibler_divergence',
'poisson',
'cosine_proximity']

for obj_fun in object_function:
    print '------------------------------------------------'
    print 'This is',str(i),'times!'
    print 'use:',obj_fun
    model = Model(sequence_input, preds)
    model.compile(loss=obj_fun,
                  optimizer='rmsprop',
                  metrics=['acc'])
    # 'recall' 'fbeta_score'  用GPU的情况不能用  为何？？
    # In[33]:

    print("model fitting - attention LSTM network")
    #model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=40, batch_size=10)
'''

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
                            
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100, return_sequences=True),merge_mode='sum')(embedded_sequences)
#l_lstm_drop = Dropout(0.3)(l_lstm)
l_att = AttLayer()(l_lstm)
#l_att_drop = Dropout(0.5)(l_att)
preds = Dense(len( model_labels[0]), activation='softmax',activity_regularizer= ActivityRegularizer(l2=0.00005))(l_att)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer= 'rmsprop',
              metrics=['acc'])
'''

# In[33]:

print("model fitting - attention LSTM network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=40, batch_size=10)

# mymodel = Sequential()
# mymodel.add( InputLayer((None,MAX_SEQUENCE_LENGTH)))
# mymodel.add( Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True))
# mymodel.add(Bidirectional(LSTM(100, return_sequences=True),merge_mode='sum'))
# mymodel.add(Dense(19))
# mymodel.add(Activation('sigmoid'))

# mymodel.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# mymodel.fit(x_train, y_train, validation_data=(x_val, y_val),nb_epoch=10, batch_size=50)

# In[46]:




# In[47]:




# In[ ]:



