from gensim.models import Word2Vec
import os

writef = open("w2v_sim_table20000.csv",'a')

readpre = open("log.txt",'r')
data = readpre.readline()
data_split = data.split("), ")
abd_list = ["'",'(',")","[",",",'1','2','3','4','5','6','7','8','9','0']

text_abd_list = [',','.','/','!',';','?',':','\n']

#abd_list = ["'",'(',")","[",",",' ']

common_list = []
commonword_list = ['of','with','at','from','into','during','including','until','against','among','throughout','despite','towards','upon','concerning','to','in','for','on','by','about','like','through','over','before','between','after','since','without','under','within','along','following','across','behind','beyond','plus','except','but','up','out','around','down','off','above','near','the','a','an']

model_name = "w2v_20000.model"
model = Word2Vec.load(model_name)


total_count =  int(sys.argv[1])

for word in data_split:
	word_purify = word
	for abd in abd_list:
		word_purify = word_purify.replace(abd,"")
	word_purify = word_purify[:-1]
	if not word_purify in commonword_list: 
		if word_purify in model.wv.vocab:
			total_count = total_count - 1
			common_list.append(word_purify)
	if total_count == 0:
		break

print(common_list)

first_line = ' / '
for word in common_list:
	first_line = first_line + ','+ word
first_line = first_line + "\n"
writef.write(first_line)




print(len(model.wv.vocab))


currentline = ""
for word1 in common_list:
	currentline = word1 + ", "
	for word2 in common_list:
		currentline = currentline + str(model.wv.similarity(word1,word2)) + ", "
	currentline = currentline[:-2]
	currentline = currentline + "\n"
	writef.write(currentline)
