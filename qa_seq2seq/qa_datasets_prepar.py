"""
将数据处理成这个形式
{
    "问题1":{
        "主体":["主体1","主体3","主体3"..],
        "问题1分词后的句子":["word1","word2","word3"...],
        "答案":"答案"
    },
    "问题2":{
        ...
    }
}
"""
import re
import jieba


def get_qa_dict():
    QA_dict = {}
    for q,a in zip(open('./datasets/python_q.txt','r').readlines(),open('./datasets/python_a.txt','r').readlines()):
       q = re.sub(r'\n|\s','',q)
       a = re.sub(r'\n|\s','',a)
       if q in QA_dict:
           QA_dict[q.strip()]["ans"].append(a.strip())
       else:
           QA_dict[q.strip()] = {}
           QA_dict[q.strip()]["ans"] = [a.strip()]
           QA_dict[q.strip()]["cuted"] = ' '.join(jieba.lcut(q.strip()))
           QA_dict[q.strip()]["entire"] = q.strip()

    return QA_dict



qa_dict = get_qa_dict()
