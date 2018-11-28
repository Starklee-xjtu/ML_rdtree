import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_read import *
import Config

wave_train = WaveDate()
wave_train.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='3')
wave_train.wave_cut()
wave_train.add_noise(-5)
wave_train.trans_norm()
wave_train.trans_fft()
x_train = wave_train.data_cut
x_train = np.squeeze(x_train)
y_train = wave_train.label

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

wave_test = WaveDate()
wave_test.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='1')
wave_test.wave_cut()
wave_test.add_noise(-5)
wave_test.trans_norm()
wave_test.trans_fft()
x_test1 = wave_train.data_cut
x_test1 = np.squeeze(x_test1)
y_test1 = wave_test.label

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predict = rfc.predict(X_test)
print(rfc.score(X_test, y_test))
predict = rfc.predict(x_test1)
print(rfc.score(x_test1, y_test1))

print('aa')

