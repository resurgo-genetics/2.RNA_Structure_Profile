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
from sklearn import metrics

train_num  = 0
test_num  = 0

kmer = int(sys.argv[1]) # the length of k-mer
swin = int(sys.argv[2]) # the length of sliding windows
vlen = int(sys.argv[3]) # the length of embedding vector

print "k:",kmer,"sliding window:",swin,"embedding vector size:",vlen
#bed2sent: convert the enhancers.bed and promoters.bed to kmer sentense
#bed format: chr start end name
def bed2sent(filename,k,win):
	#if os.path.isfile(filename+'.fa') == False:
	#	os.system("bedtools getfasta -fi /home/openness/common/igenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa -bed "+filename+".bed -fo "+filename+".fa")
	os.system("bedtools getfasta -fi /home/openness/common/igenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa -bed "+filename+".bed -fo "+filename+".fa")
	time.sleep(40)	
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
	global train_num,test_num
	fin1  = open('enhancers.bed','r')
	fin2  = open('promoters.bed','r')
	fout1 = open('enhancer.bed','w')
	fout2 = open('promoter.bed','w')
	fout3 = open('infer_enhancer.bed','w')
	fout4 = open('infer_promoter.bed','w')
	enhancers = []
	promoters = []
	infer_enhancer = []
	infer_promoter = []
	fin3 = open('testdata.txt','r')
	fout5 = open('test_index.txt','w') 
	for line in fin3:
		data = line.strip().split(',')
		test_num = test_num + 1
		if data[5] not in infer_enhancer:
                	infer_enhancer.append(data[5])
        	if data[10] not in infer_promoter:
                	infer_promoter.append(data[10])
		fout5.write(str(infer_enhancer.index(data[5]))+'\t'+str(infer_promoter.index(data[10]))+'\t'+data[7]+'\n')
	
	for line in fin1:
                data = line.strip().split()
		if data[3] not in infer_enhancer:
               		enhancers.append(data[3])
        for line in fin2:
                data = line.strip().split()
		if data[3] not in infer_promoter:
                	promoters.append(data[3])
	print len(enhancers),len(promoters),len(infer_enhancer),len(infer_promoter)
	
	fin4  = open('traindata.txt','r')
	fout6 = open('train_index.txt','w')
	for line in fin4:
		data = line.strip().split(',')
                train_num = train_num + 1
		if data[5] not in enhancers:
                        enhancers.append(data[5])
                if data[10] not in promoters:
                        promoters.append(data[10])
		fout6.write(str(enhancers.index(data[5]))+'\t'+str(promoters.index(data[10]))+'\t'+data[7]+'\n')				

	for line in enhancers:
		data  = line.split('|')
		data1 = data[1].split(':')
		data2 = data1[1].split('-')
		bed   = data1[0]+'\t'+data2[0]+'\t'+data2[1]+'\n'
		fout1.write(bed)
	for line in promoters:
		data  = line.split('|')
                data1 = data[1].split(':')
                data2 = data1[1].split('-')
                bed   = data1[0]+'\t'+data2[0]+'\t'+data2[1]+'\n'
                fout2.write(bed)
	for line in infer_enhancer:
		data  = line.split('|')
                data1 = data[1].split(':')
                data2 = data1[1].split('-')
                bed   = data1[0]+'\t'+data2[0]+'\t'+data2[1]+'\n'
                fout3.write(bed)
	for line in infer_promoter:
		data  = line.split('|')
                data1 = data[1].split(':')
                data2 = data1[1].split('-')
                bed   = data1[0]+'\t'+data2[0]+'\t'+data2[1]+'\n'
                fout4.write(bed)

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
	for epoch in range(10):
        	model.train(sentences.sentences_perm())
	model.save(name+'_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')

#train the model and print the result
def train(k,balance,vlen,swin):
	global train_num,test_num
	enhancer_model = Doc2Vec.load('enhancer_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')
	promoter_model = Doc2Vec.load('promoter_'+str(k)+'_'+str(swin)+'_'+str(vlen)+'.d2v')
	efilename  = 'infer_enhancer_'+str(k)+'_'+str(swin)+'.sent'
	pfilename  = 'infer_promoter_'+str(k)+'_'+str(swin)+'.sent'
	fine   = open(efilename,'r')
	finp   = open(pfilename,'r')
	elines = fine.readlines()
	plines = finp.readlines()
	print len(elines),len(plines)
	train_arrays = numpy.zeros((train_num, vlen*2))
	train_labels = numpy.zeros(train_num)
	test_arrays = numpy.zeros((test_num, vlen*2))
        test_labels = numpy.zeros(test_num)
	estimator = GradientBoostingClassifier(n_estimators = 8000, learning_rate = 0.001, max_depth = 25, max_features = 'log2', random_state = 0)
	fin1 = open('train_index.txt','r')
	fin2 = open('test_index.txt','r')
	i = 0
	for line in fin2:
		data = line.strip().split()
		elist = elines[int(data[0])].strip().split()
		plist = plines[int(data[1])].strip().split()
    		enhancer_vec = enhancer_model.infer_vector(elist)
    		promoter_vec = promoter_model.infer_vector(plist)
		enhancer_vec = enhancer_vec.reshape((1,vlen))
		promoter_vec = promoter_vec.reshape((1,vlen))
		test_arrays[i] = numpy.column_stack((enhancer_vec,promoter_vec))
                test_labels[i] = int(data[2])
		i = i + 1
	i = 0 
	for line in fin1:
                data = line.strip().split()
                prefix_enhancer = 'ENHANCER_' + data[0]
                prefix_promoter = 'PROMOTER_' + data[1]
		enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
                promoter_vec = promoter_model.docvecs[prefix_promoter]
                enhancer_vec = enhancer_vec.reshape((1,vlen))
                promoter_vec = promoter_vec.reshape((1,vlen))
                train_arrays[i] = numpy.column_stack((enhancer_vec,promoter_vec))
                train_labels[i] = int(data[2])
		i = i + 1
	print train_arrays[0:10,:]
	print train_labels[0:10]
	print test_arrays[0:10,:]
	estimator.fit(train_arrays,train_labels)
	predict_labels = estimator.predict(test_arrays)
	print metrics.f1_score(test_labels,predict_labels)
		

generateTraining()
print 'generate training set done!'
#bed2sent("promoter",kmer,swin)
#bed2sent("infer_promoter",kmer,swin)
#bed2sent("enhancer",kmer,swin)
#bed2sent("infer_enhancer",kmer,swin)
#print 'pre process done!'
#doc2vec("promoter",kmer,vlen)
#doc2vec("enhancer",kmer,vlen)
#print 'doc2vec done!'
print 'balanced:'
train(kmer,1,vlen,swin)
#print 'unbalanced:'
#train(kmer,0,vlen,swin)
