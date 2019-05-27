# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:05:30 2016

@author: ldy
"""
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys
import importlib
import os
importlib.reload(sys)
global category
category=['0']
os.getcwd()
print("上一级的工作目录为：%s" %os.path.abspath('..'))

path0=os.path.abspath('..')
print(path0)


# 加载文件，导入数据,分词
def loadfile():
    ori=pd.read_excel(os.path.join(path0,'data/projectinfo.xlsx'),header=None,index=None)
    ori=ori.dropna(axis=0)
    print(ori.shape)
    tempdic={'description':[]}
    global category
    cont=[]
    num=0
    for i in range(1,ori.shape[0]):
        temp1=ori.iloc[i,0]
        temp2=ori.iloc[i,1]
        temp3=ori.iloc[i,2]
        temp4=ori.iloc[i,3]
        temp0=temp1+temp2+temp3+temp4
        temp5=ori.iloc[i,6]

        for j in range(0,len(category)):
            if temp5 == category[j]:
                num = j
                break
            if j==len(category)-1:
                category.append(temp5)
                num = j+1

        cont.append(num)

        tempdic['description'].append(temp0)

    des_p=pd.DataFrame(tempdic)
    cw = lambda x: list(jieba.cut(x))
    print('0')
    des_p['word'] = des_p['description'].apply(cw)

    #print pos['words']
    #use 1 for positive sentiment, 0 for negative
    y = np.array(cont)
    print('1')
    x_train, x_test, y_train, y_test = train_test_split(np.array((des_p['description'])), y, test_size=0.2)
    print('2')
    np.save(os.path.join(path0,'svm_data/y_train.npy'),y_train)
    np.save(os.path.join(path0,'svm_data/y_test.npy'),y_test)
    print('finishload')
    return x_train,x_test
 


#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count


    return vec
    
#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=15)
    imdb_w2v.build_vocab(x_train)
    
    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train,epochs=imdb_w2v.epochs, total_examples=imdb_w2v.corpus_count)
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save(os.path.join(path0,'svm_data/train_vecs.npy'),train_vecs)
    print (train_vecs.shape)
    #Train word2vec on test tweets
    imdb_w2v.train(x_test,epochs=imdb_w2v.epochs,total_examples=imdb_w2v.corpus_count)
    imdb_w2v.save(os.path.join(path0,'svm_data/w2v_model/w2v_model.pkl'))
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save(os.path.join(path0,'svm_data/test_vecs.npy'),test_vecs)
    print (test_vecs.shape)

    print('get train vector')



def get_data():
    train_vecs=np.load(os.path.join(path0,'svm_data/train_vecs.npy'))
    y_train=np.load(os.path.join(path0,'svm_data/y_train.npy'))
    test_vecs=np.load(os.path.join(path0,'svm_data/test_vecs.npy'))
    y_test=np.load(os.path.join(path0,'svm_data/y_test.npy'))
    print('get data')
    return train_vecs,y_train,test_vecs,y_test
    

##训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, os.path.join(path0,'svm_data/svm_model/model.pkl'))
    print (clf.score(test_vecs,y_test))
    print('trainSVM')
    
    
##得到待预测单个句子的词向量    
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load(os.path.join(path0,'svm_data/w2v_model/w2v_model.pkl'))
    #imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim,imdb_w2v)
    print('get_predict_word')
    #print train_vecs.shape
    return train_vecs
    
####对单个句子进行情感判断    
def svm_predict(string):
    global category
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load(os.path.join(path0,'svm_data/svm_model/model.pkl'))
     
    result=clf.predict(words_vecs)
    print (category[int(result[0])])
    return 0

if __name__=='__main__':
    
    
    #导入文件，处理保存为向量
    x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
    get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
    train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
    svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型
    

##对输入句子情感进行判断
    string='甲醇制烯烃（二期）项目'
    svm_predict(string)
    
