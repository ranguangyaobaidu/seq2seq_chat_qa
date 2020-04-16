import config
import logging
import jieba
import jieba.posseg as psg


def jieba_cut():
    # jieba设置Log
    jieba.setLogLevel(logging.INFO)
    # 加载词典
    jieba.load_userdict(config.STOP_WORD)

    return psg