import os
import sys
sys.path.insert(0,os.getcwd())
from chat_cut_sentence import CHATCUTSENtence
import config
from utils.jieba_cut import jieba_cut
import re

jieba = jieba_cut()
cut_sentence = CHATCUTSENtence()

def prepar_through_re(content):
    content = re.sub(r'\.+','',content)
    content = re.sub(r'（.*）|「.*?」| alink|-?|\(.*?\)','',content)
    content = re.sub("我在.*?alink|alink|（.*?\d+x\d+.*?）|#|】|【|-+|_+|via.*?：*.*", " ", content)
    content = re.sub("\s+|\n", "", content)
    content = ''.join([i for i in content if i != '…' and i != ' '])

    return content


def handel_weibo(read_file,write_file,by_word=False):
    """
    处理微博数据

    """
    i = 0
    with open(read_file,'w') as writer:
        for line in open(write_file,'r'):
            sentence = prepar_through_re(line)
            if len(sentence) < 1:
                continue
            # 写入文件中
            content = cut_sentence.cut_sentence(sentence,by_word=by_word)
            writer.write(content + '\n')
            i += 1
            print('写入第{}条数据到文件中'.format(i))


def handle_xiaohuangji_chat(write_ask_file,write_response_file,by_word=True,punctuation=False):
    """
    处理小黄鸡数据
    """
    f_ask = open(write_ask_file,'a')
    f_response = open(write_response_file,'a')

    i = 0
    data = []
    for line in open(config.SMALL_YELLOW_CHICKEN):
        if line.strip() == 'E':
            if len(data) != 2:
                data = []
            continue
        if len(data) == 2:
            # 写入文件中
            ask = data[0][1:].strip()
            response = data[1][1:].strip()

            if len(ask) >0 and len(response) > 0:

                ask = cut_sentence.cut_sentence(sentence=ask,by_word=by_word,punctuation=punctuation)
                response = cut_sentence.cut_sentence(sentence=response,by_word=by_word,punctuation=punctuation)
                f_ask.write(ask + '\n')
                f_response.write(response + '\n')
                i += 1
                print('开始写入第{}对数据到文件中'.format(i))
            data = []
        data.append(line)

    f_ask.close()
    f_response.close()

if __name__ == '__main__':


    handle_xiaohuangji_chat(write_ask_file=config.CHAT_WEIBO_PRE_INPUT_PATH,
                            write_response_file=config.CHAT_WEIBO_PRE_OUTPUT_PATH,by_word=False)

    handle_xiaohuangji_chat(write_ask_file=config.CHAT_WEIBO_PRE_TINPUT_WORD_PATH,
                            write_response_file=config.CHAT_WEIBO_PRE_OUTPUT_WORD_PATH,by_word=True)

    handle_xiaohuangji_chat(write_ask_file=config.CHAT_WEIBO_PRE_TINPUT_WORD_PUNCTUATION_PATH,
                            write_response_file=config.CHAT_WEIBO_PRE_OUTPUT_WORD_PUNCTUAION_PATH,by_word=True,punctuation=True)

    # handel_weibo(read_file=config.WEIBO_CHAT_ASK_PATH,
    #              write_file=config.CHAT_WEIBO_PRE_TINPUT_WORD_PATH,
    #              by_word=True)
    # handel_weibo(read_file=config.WEIBO_CHAT_RESPONSE_PATH,
    #              write_file=config.CHAT_WEIBO_PRE_OUTPUT_PATH,
    #              by_word=False)
    # handel_weibo(read_file=config.WEIBO_CHAT_RESPONSE_PATH,
    #              write_file=config.CHAT_WEIBO_PRE_OUTPUT_WORD_PATH,
    #              by_word=True)

