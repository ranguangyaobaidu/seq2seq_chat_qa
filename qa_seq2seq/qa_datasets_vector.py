import pysparnn.cluster_index as ci
from sklearn.feature_extraction.text import TfidfVectorizer
from qa_datasets_prepar import get_qa_dict


QA_dict = get_qa_dict()

def bulit_q_vector():

    lines_cuted = [value['cuted'] for value in QA_dict.values()]
    # 实例化conunt

    tfidf = TfidfVectorizer()
    tfidf.fit(lines_cuted)

    features_vec = tfidf.transform(lines_cuted)

    return tfidf,features_vec,lines_cuted

tfidf,features_vec,lines_cuted = bulit_q_vector()
print('ljflsdfsdf')
print(tfidf.transform(['byebye']))


cp = ci.MultiClusterIndex(features_vec, lines_cuted)
ret = ['python 什么']

#对用户输入的句子进行向量化
search_vec = tfidf.transform(ret)
print(search_vec)
#搜索获取结果，返回最大的8个数据，之后根据`main_entiry`进行过滤结果
cp_search_list = cp.search(search_vec, k=8, k_clusters=10, return_distance=True)
print(cp_search_list)
