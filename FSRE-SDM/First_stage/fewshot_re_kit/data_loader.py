import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import pdb

Relations = json.load(open('./data/pid2name.json', 'r'))

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, test=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.test =test

    def __getraw__(self, item, Prompt_R=None):
        word, pos1, pos2, type_ids = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0],Prompt_R=Prompt_R)
        return word, pos1, pos2, type_ids

    def __additem__(self, d, word, type_ids, pos1, pos2, Rel_pos):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['Rel_pos'].append(Rel_pos)
        d['type_ids'].append(type_ids)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'Rel_pos': [], 'type_ids': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'Rel_pos': [], 'type_ids': []}
        query_label, support_label = [], []

        Q_na = int(self.na_rate * self.Q)
        target_classes = random.sample(self.classes, self.N)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        Prompt_R = []
        Rel_Index = []
        for i,class_name in enumerate(target_classes):
            r = i + 6
            r_s = '[unused%d]'%(r)
            r_descri = Relations[class_name][0].split(" ")

            Prompt_R.append(r_s)
            Rel_Index.append(len(Prompt_R))
            Prompt_R = Prompt_R + r_descri

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            label_temp = [0] * self.N
            label_temp[i] = 1
            for j in indices:
                word, pos1, pos2, type_ids = self.__getraw__(self.json_data[class_name][j])
                if count < self.K:
                    self.__additem__(support_set, word, type_ids, pos1, pos2, Rel_Index)
                    support_label.append(label_temp)
                else:
                    self.__additem__(query_set, word, type_ids, pos1, pos2, Rel_Index)
                    query_label.append(label_temp)
                count += 1

        # NA
        label_temp = [0] * self.N
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(list(range(len(self.json_data[cur_class]))),1, False)[0]
            word, pos1, pos2, type_ids = self.__getraw__( self.json_data[cur_class][index],Prompt_R)
            self.__additem__(query_set, word, type_ids, pos1, pos2, Rel_Index)
            query_label.append(label_temp)

        relation = []
        for class_name in target_classes:
            r = " : ".join(Relations[class_name])
            r = self.encoder.tokenizer.encode(r)
            # r = self.encoder.tokenizer.decode(r)
            relation.append(r)

        labels = support_label + query_label

        return support_set, query_set, relation, labels
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_Sup = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'Rel_pos': []}
    batch_Que = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'Rel_pos': []}
    batch_relation = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'Rel_pos': []}

    # batch_relation_examples = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    # batch_relation_examples1 = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []

    support_set, query_set, relation, labels = zip(*data)
    for i in range(len(support_set)):
        for k in ['word', 'pos1', 'pos2', 'Rel_pos']:
            batch_Sup[k] += support_set[i][k]
            batch_Que[k] += query_set[i][k]


        batch_label += labels[i]
        batch_relation['word'] += relation[i]


    Max_Sup = max([len(x) for x in batch_Sup['word']])
    Max_Que = max([len(x) for x in batch_Que['word']])
    Max_rel = max([len(x) for x in batch_relation['word']])

    batch_Sup['mask'] = [[1.0] * len(x) + [0.0] * (Max_Sup - len(x)) for x in batch_Sup['word']]
    batch_Sup['word'] = [x + [0] * (Max_Sup - len(x)) for x in batch_Sup['word']]

    batch_Que['mask'] = [[1.0] * len(x) + [0.0] * (Max_Que - len(x)) for x in batch_Que['word']]
    batch_Que['word'] = [x + [0] * (Max_Que - len(x)) for x in batch_Que['word']]

    batch_relation['mask'] = [[1.0] * len(x) + [0.0] * (Max_rel - len(x)) for x in batch_relation['word']]
    batch_relation['word'] = [x + [0] * (Max_rel - len(x)) for x in batch_relation['word']]

    for k in ['word', 'pos1', 'pos2', 'mask', 'Rel_pos']:
        batch_Sup[k] = torch.tensor(batch_Sup[k]).long()
        batch_Que[k] = torch.tensor(batch_Que[k]).long()
        batch_relation[k] = torch.tensor(batch_relation[k]).long()

    batch_label = torch.tensor(batch_label).long()
    return batch_Sup, batch_Que, batch_relation, batch_label

def get_loader(name, encoder, N, K, Q, batch_size, num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data', test =False):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, test=test)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            # num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word  = self.__getraw__(
                        self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(list(range(len(self.json_data[cur_class]))),1, False)[0]
            word = self.__getraw__(
                    self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


