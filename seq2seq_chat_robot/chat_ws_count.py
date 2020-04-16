"""
chat数据集词频保存
"""
import config
import pickle
from chat_word_seq2seq import ChatWordSeq2seq
import re
import chat_config


wordseq2seq = ChatWordSeq2seq()


for line in open(config.CHAT_WEIBO_PRE_OUTPUT_WORD_PUNCTUAION_PATH,'r'):
    line = re.sub(r'\n','',line)
    print('正在处理')
    wordseq2seq.fit(line)

for line in open(config.CHAT_WEIBO_PRE_TINPUT_WORD_PUNCTUATION_PATH,'r'):
    line = re.sub(r'\n','',line)
    print('正在处理')
    wordseq2seq.fit(line)

wordseq2seq.bulit_vocab(max_features=chat_config.MAX_FEATURES)
pickle.dump(wordseq2seq,open('./model/ws_word_punctuation.pkl','wb'))
