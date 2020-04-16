import pysparnn.cluster_index as ci
from sklearn.feature_extraction.text import TfidfVectorizer

#1. 原始数据
data = [
    'hello world',
    'oh hello there',
    'Play it',
    'Play it again Sam',
]


#2. 原始数据向量化

tv = TfidfVectorizer()
tv.fit(data)

features_vec = tv.transform(data)


# 原始数据构造索引
cp = ci.MultiClusterIndex(features_vec, data)

# 待搜索的数据向量化
search_data = [
    'oh there',
    'Play it again Frank'
]

search_features_vec = tv.transform(search_data)

#3. 索引中传入带搜索数据，返回结果
a = cp.search(search_features_vec, k=1, k_clusters=2, return_distance=False)
print(a)