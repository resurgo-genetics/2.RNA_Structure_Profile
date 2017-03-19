import os 

fin = open('result.txt','r')
result = []
for line in fin:
	result.append(float(line.strip()))

print sum(result[0:210])
print sum(result[210:])

fin = open('train_data.txt','r')
for line in fin:
	data = line.strip().split(',')
	print len(data)
