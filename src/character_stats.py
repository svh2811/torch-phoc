import numpy as np
import matplotlib.pyplot as plt


def return_hist(_set, X):
	Y = np.zeros(len(X))

	f = open('./splits/'+_set+'.txt')
	Q = f.readlines()
	f.close()
	Q = [x.rstrip('\n') for x in Q]


	for files in Q:	
		A = np.load('./annotations/current/'+files+'.npy').item()
		#print len(A)
		for i in A.keys():
			label = A[i]['name']
			for j in label:
				#print j
				if j in X:
					#print j, label
					Y[X.index(j)] = Y[X.index(j)]+1

	return Y

X = [chr(i) for i in range(ord('&'), ord('&')+1) + range(ord('A'), ord('Z')+1) + \
                    range(ord('a'), ord('z') + 1) + range(ord('0'), ord('9') + 1)]

A = return_hist('train', X)
B = return_hist('val', X)
C = return_hist('test', X)

#print map(int,C)

range_of_values = np.arange(len(A))
#ax = plt.subplot(111)
plt.bar(range_of_values+0.3, A, width=0.3, color='r', label = 'train', align='center')
plt.bar(range_of_values, B, width=0.3, color='g', label = 'val', align='center')
plt.bar(range_of_values-0.3, C, width=0.3, color='b', label = 'test', align='center')
plt.xticks(range_of_values, X)
plt.autoscale(tight=True)
plt.legend()
plt.show()