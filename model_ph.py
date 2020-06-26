import os
import sys

from gensim.models import Word2Vec

from pyphonetics import Soundex
from pyphonetics import Metaphone

def takeSecond(elem):
    return elem[1]

def get_levenshtein_distance(word1, word2):

    word2 = word2.lower()
    word1 = word1.lower()
    matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]

    for x in range(len(word1) + 1):
        matrix[x][0] = x
    for y in range(len(word2) + 1):
        matrix[0][y] = y

    for x in range(1, len(word1) + 1):
        for y in range(1, len(word2) + 1):
            if word1[x - 1] == word2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )

    return matrix[len(word1)][len(word2)]



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

vocablen = len(model.wv.vocab)
#print(vocablen)
searchno = topno * 10

for testb in testbench:
	print("Testing: ",testb)
	print("---W2V Model---")
	print(model.wv.most_similar(testb, topn=topno))
	#for i in range(vocablen):
	topword_list = []
	searchspace = model.wv.most_similar(testb, topn=searchno)
	#print(searchspace)
	for word in searchspace:
		simi = model.wv.similarity(testb,word[0])
		phoneti = get_levenshtein_distance(testb,word[0])
		#print(phoneti)
		topword_list.append([word,simi/(phoneti+5)*5])
	topword_list.sort(key=takeSecond,reverse = True)
	print("---PH Model---")
	for i in range(topno):
		print(topword_list[i])









