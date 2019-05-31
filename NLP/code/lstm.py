# -*- coding: utf-8 -*-

import yaml
import sys
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import jieba
import pandas as pd
import sys
import importlib
import os
import matplotlib.pyplot as plt
CUDA_VISIBLE_DEVICES=1

importlib.reload(sys)
global category
category=['表头']




os.getcwd()
print("上一级的工作目录为：%s" %os.path.abspath('..'))

path0=os.path.abspath('..')
print(path0)

sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 256
maxlen = 100
n_iterations = 500 # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 200
input_length = 100
cpu_count = multiprocessing.cpu_count()

def loadfile_for_test():
    ori = pd.read_excel(os.path.join(path0, 'data/projectinfo.xlsx'), header=None, index=None)
    ori = ori.dropna(axis=0)
    print(ori.shape)
    tempdic = {'description': []}
    cont0 = []
    num = 0
    for i in range(1, ori.shape[0]):
        temp1 = ori.iloc[i, 0]
        temp2 = ori.iloc[i, 1]
        temp3 = ori.iloc[i, 2]
        temp4 = ori.iloc[i, 3]
        temp10 = ori.iloc[i, 4]
        temp5 = ori.iloc[i, 6]
        temp0 = temp1 + temp2 + temp3 + temp4 + temp10
        for j in range(1, len(category)):
            if temp5 == category[j]:
                num = j
                cont0.append(temp5)
                tempdic['description'].append(temp0)
                break

    des_p = pd.DataFrame(tempdic)



    # print pos['words']
    # use 1 for positive sentiment, 0 for negative

    print('1')
    #x_train, x_test, y_train, y_test = train_test_split(np.array((des_p['description'])), y, test_size=0.2)
    combined=np.array(des_p['description'])
    cont0=np.array(cont0)

    return combined,cont0
#加载训练文件
def loadfile():
    global category
    ori = pd.read_excel(os.path.join(path0, 'data/projectinfo.xlsx'), header=None, index=None)
    ori = ori.dropna(axis=0)
    print(ori.shape)
    tempdic = {'description': []}
    cont = []
    num = 0
    for i in range(1, ori.shape[0]):
        temp1 = ori.iloc[i, 0]
        temp2 = ori.iloc[i, 1]
        temp3 = ori.iloc[i, 2]
        temp4 = ori.iloc[i, 3]
        temp10 = ori.iloc[i, 4]
        temp11 = ori.iloc[i, 5]
        temp0 = temp1+temp2+temp3+temp4+temp10
        temp5 = ori.iloc[i, 6]

        for j in range(0, len(category)):
            if temp5 == category[j]:
                num = j
                break
            if j == len(category) - 1:
                category.append(temp5)
                num = j + 1
        cont.append(num)
    print([cont.count(1),cont.count(2),cont.count(3),cont.count(4),cont.count(5),cont.count(6),cont.count(7)])
    maxnum=min([cont.count(1),cont.count(2),cont.count(3),cont.count(4),cont.count(5),cont.count(6),cont.count(7)])
    print(maxnum)
    cont=[]
    print(category)
    for i in range(1, ori.shape[0]):
        temp1 = ori.iloc[i, 0]
        temp2 = ori.iloc[i, 1]
        temp3 = ori.iloc[i, 2]
        temp4 = ori.iloc[i, 3]
        temp10 = ori.iloc[i, 4]
        temp11 = ori.iloc[i, 5]
        temp0 = temp1 + temp2 + temp3 + temp4 + temp10
        temp5 = ori.iloc[i, 6]

        for j in range(1, len(category)):
            if temp5 == category[j]and cont.count(j)<=maxnum:
                num = j
                cont.append(num)
                tempdic['description'].append(temp0)
                break

    des_p = pd.DataFrame(tempdic)
    print('0')


    # print pos['words']
    # use 1 for positive sentiment, 0 for negative
    y = np.array(cont)
    print('1')
    #x_train, x_test, y_train, y_test = train_test_split(np.array((des_p['description'])), y, test_size=0.2)
    combined=np.array(des_p['description'])
    y0 = np.zeros([len(y),len(category)-1], dtype=int)
    for i in range(0,len(y)):
        for j in range(1,len(category)):
            if y[i]==j:
                y0[i,j-1]=1
                break

    y=y0

    return combined,y

#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text



#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined,epochs=model.epochs,total_examples=model.corpus_count)
    model.save(os.path.join(path0,'lstm_data/Word2vec_model.pkl'))
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print (x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    global category
    print ('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    print (n_symbols)
    print (vocab_dim)

    model.add(LSTM(output_dim=256, activation='sigmoid'))
    # model.add(LSTM(output_dim=128, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(len(category)-1))
    model.add(Activation('sigmoid'))
    print ('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print ("Train...")
    ES = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    re_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,min_lr=0.0001)
    history=model.fit(x_train, y_train,callbacks=[re_lr,ES], batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss ')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open(os.path.join(path0,'lstm_data/lstm.yml'), 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights(os.path.join(path0,'lstm_data/lstm.h5'))
    print ('Test score:', score)


#训练模型，并保存
def train():
    print ('Loading Data...')
    combined,y=loadfile()
    print (len(combined),len(y))
    print ('Tokenising...')
    combined = tokenizer(combined)
    print ('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print ('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print (x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)




def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load(os.path.join(path0,'lstm_data/Word2vec_model.pkl'))
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    global category
    print ('loading model......')
    with open(os.path.join(path0,'lstm_data/lstm.yml'), 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print ('loading weights......')
    model.load_weights(os.path.join(path0,'lstm_data/lstm.h5'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    re=category
    result=model.predict_classes(data)
    n = result[0]+1
    print(category[int(n)])

if __name__=='__main__':
    loadfile()#请勿注释这一条会导致错误
    #train()  #这是训练，请勿轻易取消注释会重新训练

    #测试请修改loadfile_for_test中的文件名为测试用excel，excel格式和所给的训练集需格式一样！可以得到xtest和ytest如下
    xtest,ytest=loadfile_for_test() #xtest为描述向量，ytest为对应项目名称
    #写个for循环根据测试集个数依次引用numpy array类型的xtest 和 ytest 做判断即可，判断使用lstm_predict函数例子如下
    #返回值为项目名称
    print(category)
    string='年产甲醇180万吨、烯烃80万吨。'
    lstm_predict(string)
    string='西安地铁二号线从西安铁路北客站到长安韦曲，线路全长26.4公里，'
    lstm_predict(string)
    string='该电厂每年发电100MW'

    lstm_predict(string)

