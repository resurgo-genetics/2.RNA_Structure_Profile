#coding:utf-8
import pickle as pkl
import numpy as np
input_file = open('./Gen_data/14_Network_input.pkl','r')
pro_input = pkl.load(input_file)
ehr_input = pkl.load(input_file)
Y=pkl.load(input_file)
input_file.close()

print np.shape(pro_input)
print np.shape(pro_input[0])
print np.shape(ehr_input)


input_file = open('./Gen_data/14_embedding_matrix.pkl','r')
pro_embedding_matrix = pkl.load(input_file)
ehr_embedding_matrix = pkl.load(input_file)
input_file.close()

print np.shape(pro_embedding_matrix)
print np.shape(ehr_embedding_matrix)
# 切分训练集

from numpy.random import shuffle

from keras.utils.np_utils import to_categorical



SPLIT_point = int(0.85*len(Y))
seq_index = range(len(Y))
shuffle(seq_index)

Y = to_categorical(Y)
x_train ,y_train =pro_input[:SPLIT_point], Y[:SPLIT_point]
x_valid ,y_valid = pro_input[SPLIT_point:], Y[SPLIT_point:]

x_train_ehr ,x_valid_ehr =ehr_input[:SPLIT_point], ehr_input[SPLIT_point:]

print np.shape(x_train[0]),y_train[0]

# Network



from keras.layers import Embedding,InputLayer
from keras.layers import Dense,Input,Activation
from keras.layers import Embedding, LSTM, Bidirectional,GRU,InputLayer
from keras.models import Model,Sequential
from  keras.regularizers import ActivityRegularizer
from keras.layers.core import Dropout,Flatten,Merge

from keras import backend as K

# temp
import numpy as np


EMBEDDING_DIM =100
MAX_SEQUENCE_LENGTH = 800  
nb_words =4096   #字典的len(keys())

# other network

embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[pro_embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
ehr_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[pro_embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

left = Sequential()
left.add( InputLayer(input_shape=(MAX_SEQUENCE_LENGTH,),input_dtype='int32'))
left.add(embedding_layer)

right = Sequential()
right.add( InputLayer(input_shape=(MAX_SEQUENCE_LENGTH,),input_dtype='int32'))
right.add(ehr_embedding_layer)

model = Sequential()
model.add(Merge([left, right], mode='concat'))
model.add(Bidirectional(LSTM(2,return_sequences=False)))
model.add(Dense(2, activation='softmax',activity_regularizer= ActivityRegularizer(l2=0.005)))
model.compile(loss='mse', optimizer='rmsprop',metrics=['acc'])
model.summary()

model.fit([x_train, x_train_ehr], y_train, batch_size=128, nb_epoch=50, validation_data=([x_valid, x_valid_ehr], y_valid))

print 'finish!'



pro = model.predict_on_batch([x_valid,x_valid_ehr])

y_pred = [ 1 if item1>item0 else 0 for item0,item1 in pro]

from sklearn import metrics

acc = metrics.accuracy_score(y_valid[:,1],y_pred)
print acc

auc = metrics.roc_auc_score(y_valid[:,1],pro[:,1])
print auc

f1_score = metrics.f1_score(y_valid[:,1],y_pred)
print f1_score



model.save_weights('./Gen_data/14_Att-BLSTM_model.h5')
print 'save model finished!'
