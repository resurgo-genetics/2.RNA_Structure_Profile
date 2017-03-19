from sklearn.externals import joblib
import os
import random
import numpy
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

fin1  = open('all_pairs.csv','r')

pbin   = [ [],[],[],[],[] ]
nbin   = [ [],[],[],[],[] ]
bins   = []
for line in fin1:
	if line[0] == 'b':
		continue
	else:
		data  = line.split('"')
		if data[1] not in bins:
			bins.append(data[1])
		index = bins.index(data[1])
		data  = line.split(',')
		if data[7] == '1':
			pbin[index].append(line)
		else:
			nbin[index].append(line)	
train_pos = []
train_neg = []
test_pos  = []
test_neg  = []
stest_neg = []
for i in range(20):
	temp = []
	train_neg.append(temp)
	temp1 = []
	stest_neg.append(temp1)

for i in range(len(bins)):
	pbinlen = len(pbin[i])		
	nbinlen = len(nbin[i])
	print pbinlen,nbinlen
	temp1 = random.sample(pbin[i],pbinlen/10)
	temp2 = random.sample(nbin[i],nbinlen/10)
	print len(temp1),len(temp2)
	for item in temp1:
		pbin[i].remove(item)
	for item in temp2:
		nbin[i].remove(item)
	s = random.sample(nbin[i],len(nbin[i]))
	q = random.sample(temp2,len(temp2))
	test_pos.extend(temp1)
	test_neg.extend(temp2)
	train_pos.extend(pbin[i])
	nlen = len(nbin[i])/20
	qlen = len(temp2)/20
	print len(q[1*qlen:1*qlen+qlen])
	for j in range(20):
		train_neg[j].extend(s[j*nlen:j*nlen+nlen])
		stest_neg[j].extend(q[j*qlen:j*qlen+qlen])

for j in range(20):
	print len(train_pos),len(train_neg[j]),len(test_pos),len(test_neg),len(stest_neg[j])

fout1 = open('traindata.txt','w')
fout2 = open('testdata.txt','w')

for line in train_pos:
	fout1.write(line)
for line in train_neg[0]:
	fout1.write(line)

for line in test_pos:
	fout2.write(line)
for line in stest_neg[0]:
	fout2.write(line)


fin2   = open('enhancers.bed','r')
fin3   = open('promoters.bed','r')
enhancers = []
promoters = []

for line in fin2:
	data = line.strip().split()
	enhancers.append(data[3])

for line in fin3:
	data = line.strip().split()
	promoters.append(data[3])

enhancer_model = Doc2Vec.load('enhancers_6_1_100.d2v')
promoter_model = Doc2Vec.load('promoters_6_1_100.d2v')
train_num = len(train_pos) + len(train_neg[0])
test_num  = len(test_pos) + len(test_neg)
x_train = numpy.zeros((train_num, 200))
y_train = numpy.zeros(train_num)
x_test  = numpy.zeros((test_num, 200))
y_test  = numpy.zeros(test_num)

i = 0
for line in test_pos:
        data = line.strip().split(',')
        e_index = enhancers.index(data[5])
        p_index = promoters.index(data[10])
        prefix_enhancer = 'ENHANCERS_' + str(e_index)
        prefix_promoter = 'PROMOTERS_' + str(p_index)
        enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
        promoter_vec = promoter_model.docvecs[prefix_promoter]
        enhancer_vec = enhancer_vec.reshape((1,100))
        promoter_vec = promoter_vec.reshape((1,100))
	x_test[i] = numpy.column_stack((enhancer_vec,promoter_vec))
        y_test[i] = 1
        i = i + 1

for line in test_neg:
        data = line.strip().split(',')
        e_index = enhancers.index(data[5])
        p_index = promoters.index(data[10])
        prefix_enhancer = 'ENHANCERS_' + str(e_index)
        prefix_promoter = 'PROMOTERS_' + str(p_index)
        enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
        promoter_vec = promoter_model.docvecs[prefix_promoter]
        enhancer_vec = enhancer_vec.reshape((1,100))
        promoter_vec = promoter_vec.reshape((1,100))
	x_test[i] = numpy.column_stack((enhancer_vec,promoter_vec))
        y_test[i] = 0
        i = i + 1
y_pred  = numpy.zeros((20, test_num))
'''
i = 0
for line in train_pos:
        data = line.strip().split(',')
        e_index = enhancers.index(data[5])
        p_index = promoters.index(data[10])
        prefix_enhancer = 'ENHANCERS_' + str(e_index)
        prefix_promoter = 'PROMOTERS_' + str(p_index)
        enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
        promoter_vec = promoter_model.docvecs[prefix_promoter]
        enhancer_vec = enhancer_vec.reshape((1,100))
        promoter_vec = promoter_vec.reshape((1,100))
        x_train[i] = numpy.column_stack((enhancer_vec,promoter_vec))
        y_train[i] = 1
        i = i + 1

t = i
for j in range(20):
	print j
	i = t
	for line in train_neg[j]:
        	data = line.strip().split(',')
        	e_index = enhancers.index(data[5])
       		p_index = promoters.index(data[10])
        	prefix_enhancer = 'ENHANCERS_' + str(e_index)
        	prefix_promoter = 'PROMOTERS_' + str(p_index)
        	enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
        	promoter_vec = promoter_model.docvecs[prefix_promoter]
        	enhancer_vec = enhancer_vec.reshape((1,100))
        	promoter_vec = promoter_vec.reshape((1,100))
        	x_train[i] = numpy.column_stack((enhancer_vec,promoter_vec))
        	y_train[i] = 0
        	i = i + 1
	estimator = GradientBoostingClassifier(n_estimators = 8000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	estimator.fit(x_train,y_train)
	joblib.dump(estimator, "train"+str(j)+".m")
	y_pred[j,:] = estimator.predict(x_test)
'''
for j in range(20):
	estimator = joblib.load("train"+str(j)+".m")
	y_pred[j,:] = estimator.predict(x_test) 
print y_pred[:,220:260]
y_vote = numpy.zeros(test_num)
for i in range(test_num):
	y_vote[i] =  max(y_pred[:,i],key=y_pred[:,i].tolist().count)
fout0 = open('result.txt','w')
for i in range(len(y_vote)):
	fout0.write(str(y_vote[i])+'\n')

print metrics.f1_score(y_test,y_vote)

