import numpy as np
import pandas as pd
import sys
import os
os.environ['KERAS_BACKEND']='theano'
from keras.engine.topology import Layer
from keras import initializations
from keras import backend as K

from collections  import defaultdict
import pandas as pd
import numpy as np

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


#Attention GRU network
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

# Read Data

datafile  =  '/home/yinqijin/WorkSpace/2.RNA_Structure_Profile/Orig_data/pairs.csv'
import csv

with open(datafile) as csvfile:
    spamreader = csv.reader(csvfile)
    for  row in spamreader:
        csvkeys = row
        break
print csvkeys

csvdata = dict()
for item in csvkeys:
    csvdata[item] =[]
with open(datafile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Add values of every keys to the csvdata for every row
        for item in csvkeys:
            csvdata[item].append(row[item])
            if abs(int(row['enhancer_start'] )-int( row['enhancer_end'])) <6:
                print row
    

#print csvdata.keys()
for i in range(1):
    print '-'*40
    for item in csvkeys:
        print item ,': ',csvdata[item][5000]

## Make Promoter/Enhancer Sequence

from pyfasta import Fasta
genome = Fasta('/home/yinqijin/WorkSpace/DataHub/genome.fa')

#证明序列不会有错
import numpy as np
chrom = np.unique(csvdata['enhancer_chrom'])
print chrom
for item in chrom:
    #print genome[item][100:102]

#获得ｋｍｅｒ的函数
d2 = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3, 'N':0, 'n':0}
def seq_to_kspec(seq):
    mat = np.zeros((4096,1))
    k = 0
    if len(seq)<6:
        return mat
    for i in range(6):
        k = k*4 + d2[seq[i]]
    mat[k]+=1  
    for i in range(6,len(seq)):
        k = k - 4**5*d2[seq[i-6]]
        k = k*4 + d2[seq[i]]
        mat[k] += 1  
    return mat

PESeq=dict()
PESeq['Pro-index']=[]
PESeq['Pro-Seq']=[]
PESeq['Pro-Kmer']=[]
PESeq['Ehr-index']=[]
PESeq['Ehr-Seq']=[]
PESeq['Ehr-Kmer']=[]
PESeq['label'] =[]

for index in range( len( csvdata[csvkeys[0]])):
    pro_index = [csvdata['promoter_chrom'][index],csvdata['promoter_start'][index],csvdata['promoter_end'][index]]   
    ehr_index = [csvdata['enhancer_chrom'][index],csvdata['enhancer_start'][index] ,csvdata['enhancer_end'][index]]
    pro_seq =  genome[csvdata['promoter_chrom'][index]][int(csvdata['promoter_start'][index]) :int(csvdata['promoter_end'][index])].upper()
    ehr_seq =  genome[csvdata['enhancer_chrom'][index]] [int(csvdata['enhancer_start'][index]) : int(csvdata['enhancer_end'][index])].upper()    

    if len(pro_seq)<6:
        print index,'\tpro_index',pro_index
        continue
    if len(ehr_seq)<6:
        print index,'\tehr_index',ehr_index
        continue
    PESeq['Pro-index'].append(pro_index )
    PESeq['Pro-Seq'].append(pro_seq)
    PESeq['Pro-Kmer'].append( seq_to_kspec(pro_seq))
    PESeq['Ehr-index'].append(ehr_index)
    PESeq['Ehr-Seq'].append(ehr_seq)
    PESeq['Ehr-Kmer'].append(seq_to_kspec(ehr_seq))
    PESeq['label'].append(csvdata['label'][index])

print '-'*50
for item in PESeq.keys():
    print item, np.shape(PESeq[item])
print 'pos data' , sum(int( item ) for item in PESeq['label'])
print 'neg data',sum(int( 1) if item =='0' else int(0) for item in PESeq['label'])



## cut negative samples

pos_neg_index = [ int(item) for item in PESeq['label']]
print pos_neg_index[:50]
pos_neg_index = np.array(pos_neg_index )  #很关键

neg_index  = np.where(pos_neg_index==0)
pos_index  = np.where(pos_neg_index ==1)
print neg_index
print pos_index
bal_pos_neg_index = pos_index + neg_index[:len(pos_index)]
print bal_pos_neg_index[:50]
print bal_pos_neg_index[-50:]
print len(bal_pos_neg_index)

print 'lengthest Enhancer',len(max(PESeq['Ehr-Seq'],key= lambda x:len(x)))
print  'lengthest Promoter',len(max(PESeq['Pro-Seq'],key= lambda x:len(x)))

#Get X,Y with no  limited
X=[]
Y=[]
for index in range( len( PESeq['label'])):
    Y.append(PESeq['label'][index])
    X.append(np.append(PESeq['Pro-Kmer'][index],PESeq['Ehr-Kmer'][index],axis=1))
print np.shape(X)
print np.shape(Y)
X = np.array(X)
Y  = np.array(Y)

index = range(len(PESeq['label']))

# Get X,Y with   limite  making positive and negative samples 's numbers is the same
X=[]
Y=[]
for index in range( 4220):
    Y.append(PESeq['label'][index])
    X.append(np.append(PESeq['Pro-Kmer'][index],PESeq['Ehr-Kmer'][index],axis=1))
print np.shape(X)
print np.shape(Y)
X = np.array(X)
Y  = np.array(Y)


index = range(4220)

from keras.utils.np_utils import to_categorical


VALIDATION_SPLIT =0.2

np.random.seed(1234)

np.random.shuffle(index)
nb_validation_samples = int(VALIDATION_SPLIT*len(Y))


x_train = X[index[: - nb_validation_samples]]
y_train = Y[index[:-nb_validation_samples]]
x_val = X[index[-nb_validation_samples:]]
y_val =Y[index[-nb_validation_samples:]]



y_train = to_categorical(y_train) #one-hot 编码labels
y_val = to_categorical(np.asarray(y_val))

print 'x_train_shape:',np.shape(x_train)
print 'y_train_shape:',np.shape(y_train)

#使验证集正负样本数量相等
print np.shape(y_val)



# Make Network

from keras.layers import Embedding
from keras.layers import Dense,Input,Activation
from keras.layers import Embedding, LSTM, Bidirectional,GRU,InputLayer
from keras.models import Model,Sequential
from  keras.regularizers import ActivityRegularizer
from keras.layers.core import Dropout

from keras import layers

kmer_input = Input(shape=(4096,2), dtype='float32')

l_lstm =Bidirectional(LSTM(2,return_sequences=True))(kmer_input)

l_lstm_drop = Dropout(0.3)(l_lstm)
l_att = AttLayer()(l_lstm_drop)
l_att_drop = Dropout(0.5)(l_att)
preds = Dense(len( y_train[0]), activation='softmax',activity_regularizer= ActivityRegularizer(l2=0.005))(l_att)
model  = Model (kmer_input,preds)

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - attention LSTM network")
model.summary()


model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=20, batch_size=100)

model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=5, batch_size=100)

pro = model.predict_on_batch(x_val)

print pro