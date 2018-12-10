import numpy as np
from sklearn.model_selection import train_test_split
from data_read import *
import Config
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=100)
rf_per = np.empty([6,1],dtype=float)
knn_per = np.empty([6,1],dtype=float)
lg_per = np.empty([6,1],dtype=float)
rf_per_val = np.empty([6,1],dtype=float)
knn_per_val = np.empty([6,1],dtype=float)
lg_per_val = np.empty([6,1],dtype=float)
num = -1
#for nos in range(-10, 11, 4):
num = num+1
#print(nos)
nos=5
wave_train = WaveDate()
wave_train.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='1')
wave_train.wave_cut()
wave_train.add_noise(nos)
wave_train.trans_norm()
wave_train.trans_fft()
X1 = wave_train.data_cut
Y1 = wave_train.label

wave_train = WaveDate()
wave_train.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='2')
wave_train.wave_cut()
wave_train.add_noise(nos)
wave_train.trans_norm()
wave_train.trans_fft()
X2 = wave_train.data_cut
Y2 = wave_train.label

wave_train = WaveDate()
wave_train.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='0')
wave_train.wave_cut()
wave_train.add_noise(nos)
wave_train.trans_norm()
wave_train.trans_fft()
X0 = wave_train.data_cut
Y0 = wave_train.label



X = np.vstack((X1,X2,X0))
Y = np.hstack((Y1,Y2,Y0))
X = np.squeeze(X)



X=pca.fit_transform(X)
X_train, X_vali, Y_train, Y_vali = train_test_split(X, Y, test_size=0.2, random_state=42)

wave_test = WaveDate()
wave_test.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='3')
wave_test.wave_cut()
wave_test.add_noise(nos)
wave_test.trans_norm()
wave_test.trans_fft()
X_test = wave_test.data_cut
X_test = np.squeeze(X_test)
X_test=pca.transform(X_test)
Y_test = wave_test.label

# Fit an independent logistic regression model for each class using the
# OneVsRestClassifier wrapper.
base_lr = LogisticRegression(solver='lbfgs',max_iter=10000)
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_train, Y_train)

Y_pred_ovr = ovr.predict(X_vali)
ovr_jaccard_score = jaccard_similarity_score(Y_vali, Y_pred_ovr)
print (ovr_jaccard_score)

Y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_similarity_score(Y_test, Y_pred_ovr)
print (ovr_jaccard_score)

