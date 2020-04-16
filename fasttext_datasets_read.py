import config
import json
import re
from cut_sentence import CUTSentence


class FasttextDatasets(object):
    def __init__(self):
        # 问题型数据必须含有的关键词
        self.keywords_list = ["传智播客","传智","黑马程序员","黑马","python"
            "人工智能","c语言","c++","java","javaee","前端","移动开发","ui",
            "ue","大数据","软件测试","php","h5","产品经理","linux","运维","go语言",
            "区块链","影视制作","pmp","项目管理","新媒体","小程序","前端"]

        self.cut_sentence = CUTSentence()

    def judge_keyword(self,line):
        """
        判断该句子是否有关键字
        """
        for word in line.strip():
            if word in self.keywords_list:
                return True

        return False



    def handel_chicken_chat_data(self,by_word=False,path=None):
        """
        将小黄鸡数据处理为chat型数据
        """
        assert path is not None, '输入路径'
        count_num = 0
        with open(path,'a') as writer:
            with open(config.SMALL_YELLOW_CHICKEN,'r') as f:
                for line in f.readlines():

                    line = re.sub('\n',"",line)

                    if line.strip() == 'E' or len(line) < 1:
                        continue
                    elif self.judge_keyword(line): # 属于问题型数据，不做处理
                        continue
                    elif line.startswith('M'):
                        content = self.cut_sentence.cut_sentence(sentence=line[2:],by_word=by_word,with_sg=False)
                        content = " ".join(content).strip()
                        label = "\t__label__{}\n".format("chat")

                        writer.write(content + label)
                        count_num += 1
                        print('开始写入第{}条数据'.format(count_num))



    def handel_question_qa_data(self,by_word=False,path=None):
        """
        处理问题型数据
        """
        # 处理Json 读取数据
        assert path is not None,'输入路径'
        count_num = 0
        with open(path,'a') as writer:
            for value in json.load(open(config.QUESTION_JSON_PATH,'r')).values():
                for lines in value:
                    for line in lines:
                        line = re.sub(r'\n',"",line).strip()
                        content = " ".join(self.cut_sentence.cut_sentence(sentence=line,by_word=by_word,with_sg=False))
                        label = "\t__label__{}\n".format("QA")
                        writer.write(content + label)

                        count_num += 1
                        print('开始写入第{}条数据'.format(count_num))

            for line in open(config.QUESTION_TEXT_PATH,'r'):
                line = re.sub('\n',"",line).strip()
                content = self.cut_sentence.cut_sentence(sentence=line,by_word=by_word,with_sg=False)
                content = " ".join(content).strip()
                label = "\t__label__{}\n".format("QA")
                writer.write(content + label)

                count_num += 1
                print('开始写入第{}条数据'.format(count_num))



if __name__ == '__main__':
    fa = FasttextDatasets()
    # fa.handel_chicken_chat_data(path=config.FASTTEXT_CLASSIFIER_MODEL)
    # fa.handel_question_qa_data(path=config.FASTTEXT_CLASSIFIER_MODEL)
    #
    # fa.handel_chicken_chat_data(by_word=True,path=config.FASTTEXT_CLASSIFIER_WORD_DATASETS)
    # fa.handel_question_qa_data(by_word=True,path=config.FASTTEXT_CLASSIFIER_WORD_DATASETS)