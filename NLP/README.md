#环境要求
- Unix/Linux系统
- python 2.7
- python包安装： keras,sklearn,gensim,jieba,h5py,numpy,pandas，版本细节在requirements.txt有列出，但导出的我安装的pip包，有好多是没用的，如果报错请根据报错信息找对应的包修改。
```
sudo pip install -r requirements.txt
```
# 测试用法
将测试excel放在data文件中，打开/code/lstm.py 修改函数loadfile_for_test中文件名为测试excel的文件名，并在主程序部分根据提示进行测试。

#程序
- code/lstm.py 使用word2vec和LSTM训练和预测

- code/svm.py  使用word2vec和svm训练和预测

#数据
- ./data/ 原始数据文件夹
  - data/projectinfo.xls 样本原始数据

- ./svm_data/ svm数据文件夹
  - ./svm_data/\*.npy 处理后的训练数据和测试数据
  - ./svm_data/svm_model/ 保存训练好的svm模型
  - ./svm_data/w2v_model/ 保存训练好的word2vec模型


- ./lstm_data/ lstm数据文件夹
  - ./lstm_data/Word2vec_model.pkl 保存训练好的word2vec模型
  - ./lstm_data/lstm.yml  保存训练网络的结构
  - ./lstm_data/lstm.h5  保存网络训练到的权重
