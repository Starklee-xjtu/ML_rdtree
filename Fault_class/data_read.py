import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import Config as cfg
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# CLASS_NAME = {'Normal': 0,
#               'B007': 1,
#               'B014': 2,
#               'B021': 3,
#               'IR007': 4,
#               'IR014': 5,
#               'IR021': 6,
#               'OR007@6': 7,
#               'OR014@6': 8,
#               'OR021@6': 9,
#               }
#
# POSITION_NAME = ['DE_time']
#
# LOAD_NAME = '0'
#
# WINDOW_SIZE = 2048
# SAMPLE_SIZE = 256
# IMAGE_SIZE = [64, 64]

np.random.seed(666)
class WaveDate(object):
    def __init__(self, input_dir='./CWRU'):
        # 文件读取路径
        self.input_dir = input_dir

    def load_data(self, class_name, position_name, load_name):  # 用于读取文件

        # 输入参数名称
        self.class_name = class_name
        self.position_name = position_name
        self.load_name = load_name

        self.data_full = []  # 存储原始数据
        self.class_path = []  # 分析例子的路径
        self.class_label = []  # 对应的标签

        all_path = []  # 所有文件的路径

        for root, sub_folder, file_names in os.walk(self.input_dir):
            for file_name in file_names:
                if file_name.endswith('.mat') or file_name.endswith('.MAT'):
                    filepath = os.path.join(root, file_name)
                    all_path.append(filepath)  # 信号地址
                    name = file_name.split('.')[0]
                    if name[-1] == self.load_name:
                        name = name.split('_')[0]
                        if name in self.class_name:
                            self.class_path.append(filepath)
                            self.class_label.append(self.class_name[name])
        # input_data = pd.DataFrame({"path": case_path, "label": case_label})
        # input_data.to_csv(os.path.join(self.save_dir, self.choose_load + '.csv'))

        for path in self.class_path:
            datamat = sio.loadmat(path)
            names = datamat.keys()
            temp = {}
            max_len = 0
            for postion in self.position_name:
                for name in names:
                    max_len = max(max_len, np.size(datamat[name]))
                    if name[-7:] == postion:
                       temp[postion] = datamat[name].reshape(-1)    # 取出mat中的数据(由于原数据是1列排的，这里转为1行)
                if postion not in temp:
                   temp[postion] = np.zeros(max_len, np.float32)
            self.data_full.append(temp)
        return


    def wave_cut(self, window_size=cfg.WINDOW_SIZE, sample_size=cfg.SAMPLE_SIZE):  # 用于分割信号，输出时序信号

        self.window_size = window_size  # 剪切的每段信号长度，即每个样本包含的时序采样点数目
        self.sample_size = sample_size  # 期望的信号数量
        self.channel_size = len(self.position_name)  # 信号通道数目，等于测量位置数目

        self.data_cut = []       # 存储剪切后的时域波形
        self.label = []          # 存储样本标签
        for i in range(len(self.data_full)):
            wave = self.data_full[i]
            label = self.class_label[i]
            cut_data, cut_label = self.stride_way(wave, label, window_size, sample_size, 'average')
            self.data_cut.append(cut_data)
            self.label.append(cut_label)

        self.data_cut = np.array(self.data_cut).reshape(len(self.data_full) * self.sample_size, self.channel_size,
                                                        self.window_size)               # 维度变换

        self.data_cut = np.transpose(self.data_cut, (0, 2, 1)).astype(np.float32)   # [样本，时域，通道]
        self.label = np.array(self.label).flatten().astype(np.int64)
        return

    def add_noise(self, snr):  # 为每个样本添加白噪声
        self.snr = snr # 噪声分贝
        self.data_noise = self.data_cut   # 存储加噪声的时域波形
        for i in range(self.data_cut.shape[0]):  # 遍历样本
            for j in range(self.data_cut.shape[2]):  # 遍历通道
                wave = self.data_cut[i, :, j]
                Noise = self.noise(wave, snr)
                self.data_noise[i, :, j] = wave + Noise
        return

    def trans_norm(self):
        # self.data_norm = self.data_noise      # 存储归一化的时域波形
        for i in range(self.data_noise.shape[0]):   # 遍历样本
            for j in range(self.data_noise.shape[2]):   # 遍历通道
                wave = self.data_noise[i, :, j]
                self.data_noise[i, :, j] = self.norm(wave)
        return

    def norm(self, wave):
        mean = np.mean(wave)
        std = np.std(wave)
        if std <= 1e-16:
            std = 1e-16
        normwave = (wave - mean)/std
        return normwave

    # def trans_fft(self):
    #     xs = x1
    #     xf = np.fft.fft(xs)
    #     mean = np.mean(wave)
    #     std = np.std(wave)
    #     if std <= 1e-16:
    #         std = 1e-16
    #     wave_fft = (wave - mean)/std
    #     return wave_fft
    def trans_fft(self):  # 对每个样本做fft变化
        self.data_fft = self.data_cut   # 存储加噪声的时域波形
        for i in range(self.data_cut.shape[0]):  # 遍历样本
            for j in range(self.data_cut.shape[2]):  # 遍历通道
                wave = self.data_cut[i, :, j]
                temp = np.fft.fft(wave)
                self.data_fft[i, :, j] = np.abs(temp)
        return


    def finger_wave(self, image_size=cfg.IMAGE_SIZE):
        self.image_size = image_size  # 频谱图尺寸
        self.image = []  # 存储样本频谱图
        for i in range(self.data_noise.shape[0]):  # 遍历样本
            wave = self.data_noise[i, :, :]
            graph = self.finger_graph(wave, self.image_size, self.channel_size, 'fft')
            self.image.append(graph)
        self.image = np.array(self.image)  # 返回（batch，分解的频率，波段）
        return


    def stride_way(self, wave, label, window_size, sample_size, method):  # 分割信号的方法
        datas = []
        if method == 'average':  # 固定slip
            for key, value in wave.items():
                slip = int((len(value) - window_size) / (sample_size - 1))
                temp = []
                labels = []
                for i in range(sample_size):
                    temp.append(value[slip * i: (window_size + slip * i)])
                    labels.append(label)
                datas.append(temp)
        return np.array(datas).transpose((1, 0, 2)), np.array(labels)

    def noise(self, x, snr):  # 噪声
        snr = 10 ** (snr / 10.0)
        ps = np.sum(x ** 2) / len(x)
        pn = ps / snr
        return np.random.randn(len(x)) * np.sqrt(pn)

    def finger_graph(self, wave, image_size, channel_size, finger_method):
        height = image_size[0]
        width = image_size[1]
        hop_length = int(2048 / height)
        spectrogram = []
        if finger_method == 'fft':
            for i in range(channel_size):
                spectrogram.append(librosa.feature.chroma_stft(y=wave[:, i], sr=12000, n_fft=width + 1,
                                                               hop_length=hop_length, n_chroma=width))
            # 括号里的后3个参数可调，返回（频率，波段， 通道数）
            x = np.array(spectrogram).transpose((2, 1, 0))  # 在（H, W)后面加通道C。

        # if finger_method == 'emd':
        #     for i in range(channel_size):
        #
        return x.astype(np.float32)  # 因为np默认为float64，而torch的参数默认为FloatTensor，而不是DoubleTensor








if __name__ == '__main__':
    fault_name = 'IR014'
    position_name = 'DE_time'

    wave_train = WaveDate()
    wave_train.load_data(class_name={fault_name: 3}, position_name=[position_name], load_name='0')
    x1 = wave_train.data_full
    # print(x1.shape)
    x1 = np.array(x1[0][position_name])
    wave_train.wave_cut()
    wave_train.add_noise(0)
    wave_train.trans_norm()
    wave_train.trans_fft()
    x1 = wave_train.data_cut
    x1 = np.squeeze(x1)
    # wave_train.load_data(class_name={fault_name: 3}, position_name=[position_name], load_name='1')
    # x2 = wave_train.data_full
    # x2 = np.array(x2[0][position_name])
    # wave_train.load_data(class_name={fault_name: 3}, position_name=[position_name], load_name='2')
    # x3 = wa
    #     wave_train.add_noise(10)ve_train.data_full
    # x3 = np.array(x3[0][position_name])
    # wave_train.load_data(class_name={fault_name: 3}, position_name=[position_name], load_name='3')
    # x4 = wave_train.data_full
    # x4 = np.array(x4[0][position_name])

    # # log = librosa.power_to_db(sped)
    x1 = x1[10]
    sampling_rate = 12000
    plt.title('Beat wavform %s_%s'% (fault_name, position_name))

    # librosa.display.waveplot(x1, sr=sampling_rate)
    # xs = x1
    # xf=np.fft.fft(xs)
    freqs = np.fft.fftfreq(2048, 1.0/sampling_rate)
    temp1 = freqs[1:1024]
    temp2 = x1[1:1024]
    plt.subplot(4, 1, 1)
    plt.plot(freqs, x1, 'r')  # 显示原始信号的FFT模值
    plt.subplot(4, 1, 2)
    plt.plot(temp1, temp2, 'r')  # 显示原始信号的FFT模值
    plt.show()

'''
    def moving_average( data, n): #对信号做移动平均
        weights = np.ones(n)
        weights /= weights.sum()
        ma = np.convolve(data, weights, mode='full')[:len(data)]
        ma[:n] = ma[n]
        return ma
    avg_len=400
    xs_avg = moving_average(xs,avg_len)
    plt.subplot(4, 1, 3)
    librosa.display.waveplot(xs_avg, sampling_rate)
    xf_avg = np.fft.fft(xs_avg)
    freqs = np.fft.fftfreq(sampling_rate, 1.0 / sampling_rate)
    plt.subplot(4, 1, 4)
    plt.title( str(avg_len))
    temp3=freqs[1:6000]
    temp4=np.abs(xf_avg)[1:6000]
    # plt.plot(freqs, np.abs(xf_avg), 'r')  # 显示原始信号的FFT模值
    plt.plot(temp3, temp4, 'r')  # 显示原始信号的FFT模值
    # # plot mel spectrogram
    # plt.subplot(4, 1, 2)
    # librosa.display.waveplot(x2, sampling_rate)
    # plt.subplot(4, 1, 3)
    # librosa.display.waveplot(x3, sampling_rate)
    # plt.subplot(4, 1, 4)
    # librosa.display.waveplot(x4, sampling_rate)
    # librosa.display.specshow(sped, sr=12000, x_axis='time', y_axis='linear')
    # plt.title('stft spectrogram')
    plt.tight_layout()  # 保证图不重叠
    # path = 'F:\myscience\CWRU\plot\\'
    # plt.savefig(path + '%s_%s'% (fault_name, position_name))
    plt.show()

    # wave_train = WaveDate()
    # # wave_train.load_data(class_name={'Normal': 0}, position_name=[position_name], load_name='3')
    wave_train.load_data(class_name=cfg.CLASS_NAME, position_name=cfg.POSITION_NAME, load_name='3')
    # # wave_train.load_data()
    wave_train.wave_cut(window_size=cfg.WINDOW_SIZE, sample_size=cfg.SAMPLE_SIZE)
    x = wave_train.data_cut
    wave_train.add_noise(10)
    wave_train.trans_norm()
    x = wave_train.data_noise
    labels = wave_train.label
    # print(x.shape)
    # xx = np.squeeze(x)
    # print(xx.shape)
    # plt.subplot(1, 1, 1)
    # librosa.display.waveplot(xx[0], 12000)
    # plt.show()

'''