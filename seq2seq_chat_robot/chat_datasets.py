import torch
from torch.utils.data import DataLoader,Dataset
import config
import chat_config
import torch
from chat_config import ws



class CHATDatasets(Dataset):
    def __init__(self):
        self.chat_x = open(config.CHAT_WEIBO_PRE_TINPUT_WORD_PUNCTUATION_PATH,'r').readlines()
        self.chat_y = open(config.CHAT_WEIBO_PRE_OUTPUT_WORD_PUNCTUAION_PATH,'r').readlines()

        assert len(self.chat_x) == len(self.chat_y), "input和target文本的数量必须相同"

    def __getitem__(self,index):
        data = self.chat_x[index].strip().split()
        target = self.chat_y[index].strip().split()

        data_len = len(data)
        data_len = data_len if data_len <= chat_config.MAX_LEN else chat_config.MAX_LEN

        return data,target,data_len

    def __len__(self):
        return len(self.chat_x)



def collate_fn(batch):
    # 对要求的长度进行排序
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    data,target,data_len = zip(*batch)

    data = torch.LongTensor([ws.transform(i,max_len=chat_config.MAX_LEN,add_eos=False) for i in data])
    target = torch.LongTensor([ws.transform(i,max_len=chat_config.MAX_LEN,add_eos=True) for i in target])
    data_len = torch.LongTensor(data_len)


    return data,target,data_len




datasets = CHATDatasets()
data_loader = DataLoader(datasets,batch_size=128,shuffle=True,collate_fn=collate_fn,drop_last=True)


if __name__ == '__main__':
    for data,target,data_len in data_loader:
        print(data)
        print(target)
        print(data_len)
        break
