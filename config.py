# import os
# print(os.listdir('/home/datasets/text/corpus/classify'))
# print(os.listdir('/home/datasets/text/raw_chat_corpus/weibo-400w'))


# 微博chat地址
WEIBO_CHAT_ASK_PATH = '/home/datasets/text/raw_chat_corpus/weibo-400w/stc_weibo_train_post'
WEIBO_CHAT_RESPONSE_PATH = '/home/datasets/text/raw_chat_corpus/weibo-400w/stc_weibo_train_response'

# 保存处理chat数据地址
CHAT_WEIBO_PRE_INPUT_PATH = './datasets/chat_input.txt'
CHAT_WEIBO_PRE_OUTPUT_PATH = './datasets/chat_output.txt'
CHAT_WEIBO_PRE_TINPUT_WORD_PATH = './datasets/chat_input_word.txt'
CHAT_WEIBO_PRE_OUTPUT_WORD_PATH = './datasets/chat_output_word.txt'

# 保存处理chat数据地址,含有标点符号的
CHAT_WEIBO_PRE_TINPUT_WORD_PUNCTUATION_PATH = './datasets/chat_input_punctuation_word.txt'
CHAT_WEIBO_PRE_OUTPUT_WORD_PUNCTUAION_PATH = './datasets/chat_output_punctuation_word.txt'


# 停用词路径
STOP_WORD = '/home/datasets/text/corpus/stopwords.txt'

# 关键词路径
KEYBOARD_WORD = '/home/datasets/text/corpus/keywords.txt'

# 小黄鸡地址
SMALL_YELLOW_CHICKEN = '/home/datasets/text/corpus/classify/小黄鸡未分词.conv'
HANDEL_SMALL_YELLOW_CHICKEN_PATH = "./datasets/handle_small_yellow_chicken.txt"

# 构造的文本问答地址
QUESTION_JSON_PATH = '/home/datasets/text/corpus/classify/手动构造的问题.json'
QUESTION_TEXT_PATH = '/home/datasets/text/corpus/classify/爬虫抓取的问题.csv'
HANDLE_QUESTION_TEXT_PATH = './datasets/handle_question_data.txt'

# 用于fasttext的分类数据集
FASTTEXT_CLASSIFIER_DATASETS = './datasets/fasttext_classifier.txt'
FASTTEXT_CLASSIFIER_WORD_DATASETS = './datasets/fasttext_classifier_word.txt'

# 保存fasttext分类模型
FASTTEXT_CLASSIFIER_MODEL = './model/fasttext_classifie_model.pkl'
FASTTEXT_CLASSIFIER_WORD_MODEL = './model/fasttext_classifie_word_model.pkl'

# 最大长度
MAX_LEN = 10

# 批量数据大小
BATCH_SIZE = 128
