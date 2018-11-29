import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from data_read import *
import Config
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.linear_model import LogisticRegression

wave_train = WaveDate()
wave_train.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='3')
wave_train.wave_cut()
wave_train.add_noise(-5)
wave_train.trans_norm()
wave_train.trans_fft()
X = wave_train.data_cut
X = np.squeeze(X)
Y= wave_train.label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

wave_test = WaveDate()
wave_test.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='1')
wave_test.wave_cut()
wave_test.add_noise(-5)
wave_test.trans_norm()
wave_test.trans_fft()
X_test1 = wave_train.data_cut
X_test1 = np.squeeze(X_test1)
Y_test1 = wave_test.label
'''
print('random_forest_predict')
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
predict = rfc.predict(X_test)
print(rfc.score(X_test, Y_test))
predict = rfc.predict(x_test1)
print(rfc.score(x_test1, y_test1))


print('Knn_predict')
knn_c = KNeighborsClassifier()
knn_c.fit(X_train, y_train)
predict = knn_c.predict(X_test)
print(knn_c.score(X_test, y_test))
print('test 结束')
print('实际精度')
predict = knn_c.predict(x_test1)
print(knn_c.score(x_test1, y_test1))
'''

# Fit an independent logistic regression model for each class using the
# OneVsRestClassifier wrapper.
base_lr = LogisticRegression(solver='lbfgs',max_iter=1000)
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_train, y_train)
Y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_similarity_score(y_test, Y_pred_ovr)
# Fit an ensemble of logistic regression classifier chains and take the
# take the average prediction of all the chains.
chains = [ClassifierChain(base_lr, order='random', random_state=i)
          for i in range(10)]
for chain in chains:
    chain.fit(X_train, y_train)

Y_pred_chains = np.array([chain.predict(X_test) for chain in
                          chains])
chain_jaccard_scores = [jaccard_similarity_score(y_test, Y_pred_chain >= .5)
                        for Y_pred_chain in Y_pred_chains]

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_similarity_score(y_test,
                                                  Y_pred_ensemble >= .5)

model_scores = [ovr_jaccard_score] + chain_jaccard_scores
model_scores.append(ensemble_jaccard_score)

model_names = ('Independent',
               'Chain 1',
               'Chain 2',
               'Chain 3',
               'Chain 4',
               'Chain 5',
               'Chain 6',
               'Chain 7',
               'Chain 8',
               'Chain 9',
               'Chain 10',
               'Ensemble')

x_pos = np.arange(len(model_names))

# Plot the Jaccard similarity scores for the independent model, each of the
# chains, and the ensemble (note that the vertical axis on this plot does
# not begin at 0).

fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title('Classifier Chain Ensemble Performance Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation='vertical')
ax.set_ylabel('Jaccard Similarity Score')
ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
colors = ['r'] + ['b'] * len(chain_jaccard_scores) + ['g']
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()

