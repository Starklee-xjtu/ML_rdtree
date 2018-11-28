CLASS_NAME = {'Normal': 0,
              'B007': 1,
              'B014': 2,
              'B021': 3,
              'IR007': 4,
              'IR014': 5,
              'IR021': 6,
              'OR007@6': 7,
              'OR014@6': 8,
              'OR021@6': 9,
              }

POSITION_NAME = ['DE_time']

LOAD_NAME = '0'

WINDOW_SIZE = 2048
SAMPLE_SIZE = 256
IMAGE_SIZE = [64, 64]



# load 3 作为训练集，load 1 作为测试集
# 模型用SVM、随机森林、xgboost、       PCA
