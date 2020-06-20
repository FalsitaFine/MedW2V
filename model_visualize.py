from gensim.models import Word2Vec
import multiprocessing
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import sys


readpre = open("log.txt",'r')
data = readpre.readline()
data_split = data.split("), ")
abd_list = ["'",'(',")","[",",",'1','2','3','4','5','6','7','8','9','0']

text_abd_list = [',','.','/','!',';','?',':','\n']

#abd_list = ["'",'(',")","[",",",' ']

common_list = []
commonword_list = ['of','with','at','from','into','during','including','until','against','among','throughout','despite','towards','upon','concerning','to','in','for','on','by','about','like','through','over','before','between','after','since','without','under','within','along','following','across','behind','beyond','plus','except','but','up','out','around','down','off','above','near','the','a','an']

 
total_count = int(sys.argv[1])
for word in data_split:
    word_purify = word
    for abd in abd_list:
        word_purify = word_purify.replace(abd,"")
    word_purify = word_purify[:-1]
    if not word_purify in commonword_list: 
        total_count = total_count - 1
        common_list.append(word_purify)
    if total_count == 0:
        break



def tsne_plot_2d(model):
    labels = []
    tokens = []

    #visualize_num = 100
    for word in common_list:
        #if (visualize_num < 0):
        #    break
        #visualize_num = visualize_num - 1
        if(word in model.wv.vocab):
            tokens.append(model[word])
            labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000, random_state=32)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    #z = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        #z.append(value[2])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

'''
def tsne_plot_3d(model, a=1):   
    words = []
    embeddings = []
    for word in list(model.wv.vocab):
        embeddings.append(model.wv[word])
        words.append(word)
    tsne_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
    embeddings_3d = tsne_3d.fit_transform(embeddings)
    a=0.1
    label = model_name

    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title("Visualize-3D")
    plt.show()

'''

model_name = "w2v_20000.model"
model = Word2Vec.load(model_name)




tsne_plot_2d(model)
#tsne_plot_2d(model)