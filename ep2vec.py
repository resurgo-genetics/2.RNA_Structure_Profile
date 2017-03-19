# system modules
import os
import time
import sys
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# random shuffle
from random import shuffle
# numpy
import numpy
import pandas as pd 
# classifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

enhancers_num = 0
promoters_num = 0
positive_num  = 0
negative_num  = 0

kmer = int(sys.argv[1]) # the length of k-mer
swin = int(sys.argv[2]) # the length of sliding windows
vlen = int(sys.argv[3]) # the length of embedding vector

print "k:",kmer,"sliding window:",swin,"embedding vector size:",vlen
#bed2sent: convert the enhancers.bed and promoters.bed to kmer sentense
#bed format: chr start end name
def bed2sent(filename,k,win):
	if os.path.isfile(filename+'.fa') == False:
		os.system("bedtools getfasta -fi /home/openness/common/igenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa -bed "+filename+".bed -fo "+filename+".fa")
		time.sleep(30)	
	fin   = open(filename+'.fa','r')
	fout  = open(filename+'_'+str(k)+'_'+str(swin)+'.sent','w')
	for line in fin:
		if line[0] =='>':
			continue
		else:
			line   = line.strip().lower()
			length = len(line)
			i = 0
			while i<= length-k:
				fout.write(line[i:i+k]+' ')
				i = i + win
			fout.write('\n')
	
#generateTraining: extract the training set from pairs.csv and output the training pair with sentence
def generateTraining():
	global enhancers_num,promoters_num,positive_num,negative_num
	fin1 = open('enhancers.bed','r')
	fin2 = open('promoters.bed','r')
	enhancers = []
	promoters = []
	for line in fin1:
		data = line.strip().split()
		enhancers.append(data[3])
		enhancers_num = enhancers_num + 1
	for line in fin2:
		data = line.strip().split()
		promoters.append(data[3])
		promoters_num = promoters_num + 1
	fin3 = open('pairs.csv','r')
	fout = open('training.txt','w')
	for line in fin3:
		if line[0] == 'b':
			continue
		else:
			data = line.strip().split(',')
			enhancer_index = enhancers.index(data[5])
			promoter_index = promoters.index(data[10])
			fout.write(str(enhancer_index)+'\t'+str(promoter_index)+'\t'+data[7]+'\n')
			if data[7] == '1':
				positive_num = positive_num + 1
			else:
				negative_num = negative_num + 1

# convert the sentence to doc2vec's tagged sentence
class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences

#train the embedding vector from the file
def doc2vec(name,k,vlen):
	filename  = name+'_'+str(k)+'_'+str(swin)+'.sent'
	indexname = name.upper()
	sources = {filename:indexname}
	sentences = TaggedLineSentence(sources)
	model = Doc2Vec(min_count=1, window=10, size=vlen, sample=1e-4, negative=5, workers=8)
	model.build_vocab(sentences.to_array())
	for epoch in range(20):
        	model.train(sentences.sentences_perm())
	model.save(name+'_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')

#train the model and print the result
def train(k,balance,vlen):
	global enhancers_num,promoters_num,positive_num,negative_num
	enhancer_model = Doc2Vec.load('enhancers_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')
	promoter_model = Doc2Vec.load('promoters_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')
	print positive_num,negative_num
	if balance == 1:
		arrays = numpy.zeros((positive_num*2, vlen*2))
		labels = numpy.zeros(positive_num*2)
		num    = positive_num*2
		estimator = GradientBoostingClassifier(n_estimators = 8000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	else:	
		arrays = numpy.zeros((positive_num+negative_num, vlen*2))
                labels = numpy.zeros(positive_num+negative_num)
		num    = positive_num+negative_num
		estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.1, max_depth = 5, max_features = 'log2', random_state = 0)
	fin = open('training.txt','r')
	i = 0
	for line in fin:
		data = line.strip().split()
		prefix_enhancer = 'ENHANCERS_' + data[0]
		prefix_promoter = 'PROMOTERS_' + data[1]
    		enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
    		promoter_vec = promoter_model.docvecs[prefix_promoter]
		enhancer_vec = enhancer_vec.reshape((1,vlen))
		promoter_vec = promoter_vec.reshape((1,vlen))
		arrays[i] = numpy.column_stack((enhancer_vec,promoter_vec))
                labels[i] = int(data[2])
		i = i + 1
		if i >=num:
			break
	cv = StratifiedKFold(y = labels, n_folds = 10, shuffle = True, random_state = 0)
        scores = cross_val_score(estimator, arrays, labels, scoring = 'f1', cv = cv, n_jobs = -1)
        print('f1:')
        print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
        scores = cross_val_score(estimator, arrays, labels, scoring = 'roc_auc', cv = cv, n_jobs = -1)
        print('auc:')
        print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
'''
	nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window', 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

	training_df = pd.read_hdf('training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
	predictors_df = training_df.drop(nonpredictors, axis = 1)
	predictors_df = predictors_df.as_matrix()	
	arrays = numpy.column_stack((arrays,predictors_df[0:num,:]))
	print arrays.shape
	cv = StratifiedKFold(y = labels, n_folds = 10, shuffle = True, random_state = 0)
	scores = cross_val_score(estimator, arrays, labels, scoring = 'f1', cv = cv, n_jobs = -1)
	print('f1:')
	print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
	scores = cross_val_score(estimator, arrays, labels, scoring = 'roc_auc', cv = cv, n_jobs = -1)
	print('auc:')
	print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
'''
'''
nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window', 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']
training_df = pd.read_hdf('training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
predictors_df = training_df.drop(nonpredictors, axis = 1)
predictors_df = predictors_df.as_matrix()
print predictors_df[0:2,1]
print predictors_df.shape
'''
bed2sent("promoters",kmer,swin)
bed2sent("enhancers",kmer,swin)
print 'pre process done!'
generateTraining()
print 'generate training set done!'
doc2vec("promoters",kmer,vlen)
doc2vec("enhancers",kmer,vlen)
print 'doc2vec done!'
print 'balanced:'
train(kmer,1,vlen)
#print 'unbalanced:'
#train(kmer,0,vlen)
