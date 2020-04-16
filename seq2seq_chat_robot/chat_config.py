import pickle

# 定义chat模型相关参数
MAX_FEATURES = 5000
MAX_LEN = 20

# 定义encoder的相关参数
ENCODER_DIM = 200
METHOD = 'concat'
HIDDEN_SIZE = 64
BATCH_SIZE = 128
DROPOUT = 0.5
NUM_LAYERS = 1



ws = pickle.load(open('./model/ws_word_punctuation.pkl','rb'))