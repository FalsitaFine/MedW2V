import os
import sys

from gensim.models import Word2Vec


model_name = "w2v_20000.model"
model = Word2Vec.load(model_name)



#testbench = ['fever','virus','dead','cancer','hospital']
testbench = []
topno = 1
for i in range(len(sys.argv)):
	if (i == 1):
		topno = int(sys.argv[i])
	if (i > 1):
		testbench.append(sys.argv[i])

for testb in testbench:
	print("Testing: ",testb)
	print(model.wv.most_similar(testb, topn=topno))







