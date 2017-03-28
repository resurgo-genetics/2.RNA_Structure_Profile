
# coding: utf-8

# In[1]:

import hickle as hkl


# In[2]:

seq_label=hkl.load('./Gen_data/15_label.hkl')
seqa_org =hkl.load('./Gen_data/15_A0.hkl')
seqb_org= hkl.load('./Gen_data/15_B0.hkl')


# In[3]:

print seqa_org[:2]


# In[4]:

#获得ｋｍｅｒ的函数
embed_dict =[]
def seq_to_embed_seq(seq):
    kmer=6
    embed_seq = []
    if len (seq)< kmer:
        return embed_seq
    for i in range(kmer,len(seq)):
        word  = seq[(i-kmer):i] 
        #if  word not in embed_dict.keys():
        #    embed_dict[ word  ] =str( len(embed_dict.keys()) ) #给标号
        #embed_seq.append(  str( embed_dict[ word  ]  )  )
        embed_seq.append(  word )
        if word not in embed_dict:
            embed_dict.append(word)
    return embed_seq


# In[ ]:

seqa_embed=[]
for seq in seqa_org:
    seq = seq.upper()
    seqa_embed.append( seq_to_embed_seq(seq))


# In[ ]:

seqb_embed=[]
for seq in seqb_org:
    seq = seq.upper()
    seqb_embed.append( seq_to_embed_seq(seq))


# # 切分训练集

# In[ ]:

from numpy.random import shuffle
import numpy as np
np.random.seed(1234)


# In[ ]:

seq_index = range(len(seq_label))
shuffle(seq_index)


# In[ ]:

split_point=int(0.8*len(seq_label))
train_index = seq_index[: split_point]
valid_index = seq_index[split_point:]


# In[ ]:




# # Embedding matrix

# In[ ]:

from gensim.models import Word2Vec,word2vec


# In[ ]:

sentenses_to_train_embeding_matrix =  np.array(seqa_embed)[train_index]
with open('./Gen_data/hic/15_seqa_sentense.txt','w') as f:
    for row in range(np.shape( sentenses_to_train_embeding_matrix)[0]):
        Sentense = sentenses_to_train_embeding_matrix[row]
        f.write( ' '.join(Sentense)+'\n')


# In[ ]:

sentenses_to_train_embeding_matrix =  np.array(seqb_embed)[train_index]
with open('./Gen_data/hic/15_seqb_sentense.txt','w') as f:
    for row in range(np.shape( sentenses_to_train_embeding_matrix)[0]):
        Sentense = sentenses_to_train_embeding_matrix[row]
        f.write( ' '.join(Sentense)+'\n')


# In[ ]:

sentences = word2vec.LineSentence('./Gen_data/hic/15_seqa_sentense.txt')
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
print(model)
#print(model.wv['AAAAAA'])
seqa_word_vectors = model.wv


# In[ ]:

sentences = word2vec.LineSentence('./Gen_data/hic/15_seqb_sentense.txt')
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
print(model)
#print(model.wv['AAAAAA'])
seqb_word_vectors = model.wv


# In[ ]:

embed_dict_len = len(embed_dict)
seqa_embedding_matrix = np.random.random((embed_dict_len,100))  #4096个字,每个字100维
for item in range(embed_dict_len):
    word  = embed_dict[item] 
    if word in seqa_word_vectors.index2word :   
        seqa_embedding_matrix[item]= seqa_word_vectors[word]


# In[ ]:

seqb_embedding_matrix = np.random.random((embed_dict_len,100))  #4096个字,每个字100维
for item in range(embed_dict_len):
    word  = embed_dict[item] 
    if word in seqb_word_vectors.index2word :   
        seqb_embedding_matrix[item]= seqb_word_vectors[word]


# In[ ]:

# 准备输入


# In[ ]:

seqa_input=[]
for item in range(len(seq_label)):
    origin_sentense = seqa_embed[item]
    index_sentense = []
    for word in  origin_sentense:      
        index_sentense.append( embed_dict.index(word))
    seqa_input.append(index_sentense)
    #print pro_index_sentense
print 'seqa_fin!'


# In[ ]:

seqb_input=[]
for item in range(len(seq_label)):
    origin_sentense = seqb_embed[item]
    index_sentense = []
    for word in  origin_sentense:      
        index_sentense.append( embed_dict.index(word))
    seqb_input.append(index_sentense)
    #print pro_index_sentense
print 'seqb_fin!'


# In[ ]:

from keras.utils.np_utils import to_categorical


# In[ ]:

Y = to_categorical(seq_label)


# In[ ]:

x_train ,y_train =seqa_input[train_index], Y[train_index]
x_valid ,y_valid = seqa_input[valid_index], Y[valid_index]

x_train_seqb, x_valid_seqb = seqb_input[train_index],seqb_input[valid_index]



# In[ ]:

import pickle as pkl
input_file = open('./Gen_data/15_data.pkl','w')
pkl.dump(x_train,input_file)
pkl.dump(x_valid,input_file)
pkl.dump(x_train_seqb,input_file)
pkl.dump(x_valid_seqb,input_file)
pkl.dump(y_train,input_file)
pkl.dump(y_valid,input_file)
pkl.dump(train_index,input_file)
pkl.dump(valid_index,input_file)
pkl.dump(embedding_matrix,input_file)
input_file.close()


# In[ ]:

import pickle as pkl
output_file = open('./Gen_data/15_data.pkl','r')
x_train = pkl.load(output_file)
x_valid = pkl.load(output_file)
x_train_seqb = pkl.load(output_file)
y_valid_seqb = pkl.load(output_file)
y_trian = pkl.load(output_file)
y_valid = pkl.load(output_file)
train_index = pkl.load(output_file)
valid_index = pkl.load(output_file)
embedding_matrix  = pkl.load(output_file)
output_file.close()


# # network

# In[ ]:

from keras.layers import Embedding,InputLayer
from keras.layers import Dense,Input,Activation
from keras.layers import Embedding, LSTM, Bidirectional,GRU,InputLayer
from keras.models import Model,Sequential
from  keras.regularizers import ActivityRegularizer
from keras.layers.core import Dropout,Flatten,Merge
from keras import backend as K
from keras import optimizers as opt


# In[ ]:

EMBEDDING_DIM =100
MAX_SEQUENCE_LENGTH = 1000  
nb_words =embed_dict_len   #字典的len(keys())


# In[ ]:

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
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


# In[ ]:

seqa_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[seqa_embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
seqb_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[seqb_embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


# In[ ]:

left = Sequential()
left.add( InputLayer(input_shape=(MAX_SEQUENCE_LENGTH,),input_dtype='int32'))
left.add(seqa_embedding_layer)

right = Sequential()
right.add( InputLayer(input_shape=(MAX_SEQUENCE_LENGTH,),input_dtype='int32'))
right.add(seqb_embedding_layer)

model = Sequential()
model.add(Merge([left, right], mode='concat'))
model.add(Bidirectional(LSTM(100,return_sequences=True)))
model.add(AttLayer())
model.add(Dense(2, activation='softmax',activity_regularizer= ActivityRegularizer(l2=0.005,l1=0.005)))
rmsprop = opt.rmsprop(lr=0.0001)
model.compile(loss='mse', optimizer=rmsprop,metrics=['acc'])
model.summary()


# In[ ]:

from sklearn import metrics

with open('./Gen_data/'+'hic'+'/15_newwork_result.txt','w') as f:
    f.write( ','.join(  ['acc','auc','f1_score']))
    for iter_index in range(60):
        model.fit([x_train, x_train_seqb], y_train, batch_size=128, nb_epoch=10, validation_data=([x_valid, x_valid_seqb], y_valid))
        pro = model.predict_on_batch([x_valid,x_valid_ehr])
        y_pred = [ 1 if item1>item0 else 0 for item0,item1 in pro]
        acc = metrics.accuracy_score(y_valid[:,1],y_pred)
        auc = metrics.roc_auc_score(y_valid[:,1],pro[:,1])
        f1_score = metrics.f1_score(y_valid[:,1],y_pred)
        f.write('{:2f},{:2f},{:2f}'.format(acc,auc,f1_score))
        print '{:2f},{:2f},{:2f}'.format(acc,auc,f1_score)
        if  iter_index%19==0 :
            model.save_weights('./Gen_data/'+'hic'+'/15_Att-BLSTM_model_iter{:03d}.h5'.format (iter_index*10))
    model.save_weights('./Gen_data/'+'hic'+'/15_Att-BLSTM_model_iter{:s}.h5'.format ('fin'))
