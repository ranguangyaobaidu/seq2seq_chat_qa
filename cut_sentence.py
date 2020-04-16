import logging
import string
import jieba
import jieba.posseg as psg
import config
import re


# jieba设置Log
jieba.setLogLevel(logging.INFO)
# 加载词典
# jieba.load_userdict(config.STOP_WORD)
# 处理英文部分
letters = string.ascii_letters
print(letters)


class CUTSentence(object):
    """
    处理分词
    """
    def __init__(self):
        self.stop_word = set([i.strip() for i in open(config.STOP_WORD,'r').readlines()]) # 停用词
        self.filters= [",","-","."," ",'{','}','[',']'] #单字分割 去除的标点


    def _cut_by_word(self,sentence):
        """
        进行单个词切分
        """
        sentence = sentence.strip()
        temp = ""
        result = []
        for word in sentence:
            if word in self.filters:
                if temp != "":
                    result.append(temp)
                    temp = ''
                continue

            if word in letters:
                # 中文中含有英文，不处理
                temp += word
            else:

                if temp != "":  # 不是英文
                    result.append(temp)
                    temp = ""
                result.append(word)

        if not result and temp != "":
            return sentence.split()

        if temp != "":  # 对最后未加入列表的数据进行加入
            result.append(temp)

        return result



    def cut_sentence(self,sentence,by_word=True,use_stop=True,with_sg=False):
        """
        with_sg : 有无词性
        """
        assert by_word != True or with_sg != True,'单个字分词没有词性'
        if by_word is True:
            return self._cut_by_word(sentence)

        else:
            jb_content = psg.lcut(sentence)

            if use_stop is True:
                # 判断是否存在停用词
                jb_content = [i for i in jb_content if i.word not in self.stop_word]
            if with_sg is True:
                jb_content = [(i.word,i.flag) for i in jb_content]

            else:
                jb_content = [i.word for i in jb_content]

        return jb_content

if __name__ == '__main__':
    cut = CUTSentence()
    print(cut.cut_sentence('我是谁 非非  诉讼费 hello world '))
