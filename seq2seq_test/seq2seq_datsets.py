import os
import sys
sys.path.insert(0,os.getcwd())

from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
import config
from num_sentence import NUMSentence


class NUMADDDatasets(Dataset):
    def __init__(self):
        self.total_data_size = 500000
        np.random.seed(10)
        self.data = np.random.randint(1,100000000,size=[self.total_data_size])

    def __getitem__(self, index):
        content = str(self.data[index])

        return content,content + '0',len(content),len(content) + 1

    def __len__(self):
        return self.total_data_size

num_sequence = NUMSentence()
def collate_fn(batch):
    # 对结果进行降序排序
    batch = sorted(batch,key=lambda x:x[3],reverse=True)

    data, label, data_lengths, label_lengths = list(zip(*batch))

    input = torch.LongTensor([num_sequence.transform(i, max_len=config.MAX_LEN) for i in data])
    target = torch.LongTensor([num_sequence.transform(i, max_len=config.MAX_LEN, add_eos=True) for i in label])
    input_length = torch.LongTensor(data_lengths)
    target_length = torch.LongTensor(label_lengths)

    return input,target,input_length,target_length



datasets = NUMADDDatasets()
data_loader = DataLoader(datasets,batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=collate_fn,drop_last=True)




if __name__ == '__main__':
    for data,target,data_len,target_len in data_loader:
        print(data)
        print(data_len)
        print(target)
        break