# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:54:57 2018

@author: TsungYuan
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gensim
import pickle
from gensim.models import Word2Vec
import nltk
from nltk import word_tokenize


bookname = ["The Passing of Empire","Patriotic Song","Latin American Mythology","The Adventures of Tom Sawyer","North of Boston","The Wonderful Wizard of Oz",
            "Forge of Foxenby","Heart of Darkness","A Tale of Two Cities","Alice’s Adventures in Wonderland","The Iliad of Homer","Democracy in America","Bulfinch's Mythology","Grimms’ Fairy Tales",
            "The Republic","The Adventures of Sherlock Holmes","Great Expectations","Adventures of Huckleberry Finn","Moby Dick; or The Whale","The Strange Case Of Dr. Jekyll And Mr. Hyde",
            "Treasure Island","On Liberty","Autobiography of Benjamin Franklin","The Spy","North and South","English Fairy Tales","Mr. Spaceship","Japanese Fairy Tales","King Arthur and the Knights of the Round Table",
            "A Critical History of Greek Philosophy","The Adventures of Sherlock Holmes","Myths of the Cherokee","A Short History of the World","Bushido, the Soul of Japan","Religion and Morality Vindicated against Hypocrisy and Pollution",
            "Pride and Prejudice","The Religion of the Samurai","Bible Myths and their Parallels in other Religions","The Children's Bible","The Religion of Ancient Rome","Religions of Ancient China",
            "Religion and Art in Ancient Greece","Religion in Japan","Ancient Egypt","The Raven","The Engineer's Sketch-Book","The Mechanical Properties of Wood","The Principles of Scientific Management",
            "The Ethical Engineer","The Devil's Dictionary","The War of the Worlds","New Atlantis","The Secret Garden","The Blue Fairy Book",
            "American Architecture; Studies","Famous Modern Ghost Stories","Irish Fairy Tales","Mark Twain's Speeches","An Account of Egypt","The Life and Adventures of Santa Claus"] 
dictname = ["The Passing of Empire","Patriotic Song","Latin American Mythology","The Adventures of Tom Sawyer","North of Boston","The Wonderful Wizard of Oz",
            "Forge of Foxenby","Heart of Darkness","A Tale of Two Cities","Alice’s Adventures in Wonderland","The Iliad of Homer","Democracy in America","Bulfinch's Mythology","Grimms’ Fairy Tales",
            "The Republic","The Adventures of Sherlock Holmes","Great Expectations","Adventures of Huckleberry Finn","Moby Dick; or The Whale","The Strange Case Of Dr. Jekyll And Mr. Hyde",
            "Treasure Island","On Liberty","Autobiography of Benjamin Franklin","The Spy","North and South","English Fairy Tales","Mr. Spaceship","Japanese Fairy Tales","King Arthur and the Knights of the Round Table",
            "A Critical History of Greek Philosophy","The Adventures of Sherlock Holmes","Myths of the Cherokee","A Short History of the World","Bushido, the Soul of Japan","Religion and Morality Vindicated against Hypocrisy and Pollution",
            "Pride and Prejudice","The Religion of the Samurai","Bible Myths and their Parallels in other Religions","The Children's Bible","The Religion of Ancient Rome","Religions of Ancient China",
            "Religion and Art in Ancient Greece","Religion in Japan","Ancient Egypt","The Raven","The Engineer's Sketch-Book","The Mechanical Properties of Wood","The Principles of Scientific Management",
            "The Ethical Engineer","The Devil's Dictionary","The War of the Worlds","New Atlantis","The Secret Garden","The Blue Fairy Book",
            "American Architecture; Studies","Famous Modern Ghost Stories","Irish Fairy Tales","Mark Twain's Speeches","An Account of Egypt","The Life and Adventures of Santa Claus"]


corpus = []

for i in range(60):
    f = open(str(i+1)+'.txt', 'r', encoding='UTF-8')
    sentence = f.read()
    f.close()
    sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    article = ""
    for word,pos in pos_tags:
         if (pos == 'NN' or pos == 'NNP'):
             article = article + " " + word
    corpus.append(article)
    print("---------"+str(i)+"---------")

transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidf.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(corpus))

word=loaded_vec.get_feature_names()#獲得詞模型中的所有詞語
weight=tfidf.toarray()#將tf-idf矩陣抽取出来，元素a[i][j]表示j词在i文本中的tf-idf權重

tfidfb = transformer.fit_transform(loaded_vec.fit_transform(corpus))
wordb=loaded_vec.get_feature_names()#獲得詞模型中的所有詞語
weightb=tfidf.toarray()#將tf-idf矩陣抽取出来，元素a[i][j]表示j词在i文本中的tf-idf權重

for i in range(len(weight)):
    print (u"-------" + bookname[i] + u" 的tf-idf權重前5大權重------")
    j = weight[i].argsort()[-5:]
    for k in range(5):
        print(word[j[4-k]],weight[i,j[4-k]])

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
#model = Word2Vec.load("word2vec.model")
weights = [0.5,0.25,0.15,0.1]

for b in range(60): 
    similar = np.zeros((60))
    for i in range(60):
        for j in range(4):
            try:
                similar[i] += model.similarity(word[weight[b].argsort()[-5:][4]], word[weight[i].argsort()[-5:][4-j]])*weights[j]*1
                #print(word[weight[b].argsort()[-5:][4]],word[weight[i].argsort()[-5:][4-j]])
            except KeyError:
                #print(str(word[weight[b].argsort()[-5:][4]])+" or "+str( word[weight[i].argsort()[-5:][4-j]])+"is not in the word")
                pass
        for j in range(4):
            try:
                similar[i] += model.similarity(word[weight[b].argsort()[-5:][3]], word[weight[i].argsort()[-5:][4-j]])*weights[j]*0.8
                #print(word[weight[b].argsort()[-5:][3]],word[weight[i].argsort()[-5:][4-j]])
            except KeyError:
                #print(str(word[weight[b].argsort()[-5:][3]])+" or "+str( word[weight[i].argsort()[-5:][4-j]])+"is not in the word")
                pass
        for j in range(4):
            try:
                similar[i] += model.similarity(word[weight[b].argsort()[-5:][2]], word[weight[i].argsort()[-5:][4-j]])*weights[j]*0.6
                #print(word[weight[b].argsort()[-5:][3]],word[weight[i].argsort()[-5:][4-j]])
            except KeyError:
                #print(str(word[weight[b].argsort()[-5:][3]])+" or "+str( word[weight[i].argsort()[-5:][4-j]])+"is not in the word")
                pass
        for j in range(4):
            try:
                similar[i] += model.similarity(word[weight[b].argsort()[-5:][1]], word[weight[i].argsort()[-5:][4-j]])*weights[j]*0.4
                #print(word[weight[b].argsort()[-5:][3]],word[weight[i].argsort()[-5:][4-j]])
            except KeyError:
                #print(str(word[weight[b].argsort()[-5:][3]])+" or "+str( word[weight[i].argsort()[-5:][4-j]])+"is not in the word")
                pass
    list_a = similar.tolist()
    list_a[b] = 0
    max_index = list_a.index(max(list_a))        
    print(str(bookname[b]) + "推薦" + str(bookname[max_index]) + " : " + str(np.max(list_a)))
        
    gindex = ""
    for group in range(60):
        if list_a[group] > 0.6:
            gindex = gindex+bookname[group]+","
    dictname[b]  = {}
    dictname[b].update({"most_relevant" : str(bookname[max_index]),"max_score" : np.max(list_a),"group" : gindex})    
