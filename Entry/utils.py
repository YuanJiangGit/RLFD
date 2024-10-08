import os
from collections import deque
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd


class Dictionary(object):
    def __init__(self):
        self.state2idx = {}
        self.id2oState = {}
        self.idx2state = []

    def add_state(self, statement):
        replaceStatement = statement.replace(' ', '')
        if replaceStatement not in self.state2idx:
            self.idx2state.append(replaceStatement)
            self.state2idx[replaceStatement] = len(self.idx2state) - 1
            self.id2oState[len(self.idx2state) - 1] = statement

        return self.state2idx[replaceStatement]

    def __len__(self):
        return len(self.idx2state)


class Corpus(object):
    def __init__(self, datasets, config):
        self.dictionary = Dictionary()
        self.dictionary.add_state('PAD')
        for dataName in datasets:
            if dataName == 'train':
                data = pd.read_csv(config.train_path, header=0, sep=',')
                self.tokenize(data.code)
            elif dataName == 'test':
                data = pd.read_csv(config.test_path, header=0, sep=',')
                self.tokenize(data.code)
            elif dataName == 'val':
                data = pd.read_csv(config.val_path, header=0, sep=',')
                self.tokenize(data.code)

    def tokenize(self, data):
        for program in data:
            for statement in eval(program):
                # statement=statement.replace(' ','')
                self.dictionary.add_state(statement)
        return True


class PgDatasetProcessing(Dataset):
    def __init__(self, config, tag='train'):
        if tag == 'train':
            path = config.train_path
        elif tag == 'test':
            path = config.test_path
        elif tag == 'val':
            path = config.val_path
        else:
            print('please input the correct dataset name!')
            path = ''
            exit()
        data = pd.read_csv(path, header=0, sep=',')
        self.programs = data.code
        self.label = data.label
        self.corpus = config.corpus
        self.sen_len = config.sentence_len

    def __getitem__(self, index):
        count = 0
        statIDs = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))

        for statement in eval(self.programs[index]):
            statement = statement.replace(' ', '')
            if statement in self.corpus.dictionary.state2idx:
                if count > self.sen_len - 1:
                    break
                statIDs[count] = self.corpus.dictionary.state2idx[statement]
                count += 1

        label = torch.LongTensor([self.label[index]])
        return statIDs, label

    def __len__(self):
        return len(self.programs)


class RinputData(Dataset):
    '''
    对产生的trajectory进行保存
    '''

    def __init__(self, maxlen=1000):
        self.RinputList = deque(maxlen=maxlen)
        self.labelList = deque(maxlen=maxlen)

    def append(self, RinputList, labelList):
        self.RinputList = self.RinputList + deque(RinputList)
        self.labelList = self.labelList + deque(labelList)

    def __getitem__(self, index):
        return self.RinputList[index], self.labelList[index]

    def __len__(self):
        return len(self.labelList)


class TempData(Dataset):
    '''
    对产生的trajectory进行保存
    '''

    def __init__(self, maxlen=1000):
        self.RinputList = deque(maxlen=maxlen)

    def append(self, RinputList):
        self.RinputList = self.RinputList + deque(RinputList)

    def __getitem__(self, index):
        return self.RinputList[index]

    def __len__(self):
        return len(self.RinputList)


class ReTrainCritic(Dataset):
    '''
    基于critic生成用于再训练的数据集
    '''

    def __init__(self, config):
        self.originInputs = []
        self.labels = []
        self.actorModel = None
        self.criticModel = None
        self.config = config

    def _update_model(self, actorModel, criticModel):
        '''
        更新模型
        :param actorModel:
        :param criticModel:
        :return:
        '''
        self.actorModel = actorModel
        self.criticModel = criticModel

    def append(self, input, label):
        '''
        添加数据
        :param input:
        :param label:
        :return:
        '''
        self.originInputs.append(input)
        self.labels.append(label)

    def _clear(self):
        self.originInputs = []
        self.labels = []
        self.actorModel = None
        self.criticModel = None

    def Sampling_RL(self, actor, critic, input, vector, length):
        '''
        :param actor:
        :param critic:
        :param input: a program (type: list,size: 3),tensor类型
        :param vector: the embedding (1 * length*embedding matrix) corresponding to the sentence
        :param length: the length of the lines of the program
        :return:
        '''
        num_layers = self.config.lstm_num_layers
        num_directions = 2
        current_lower_state = torch.zeros(1, num_layers * num_directions * self.config.lstm_hidden_dim).cuda()
        actions = []
        states = []
        # new input derived by actor and the former of new input is the same as the input
        input_clean = [[], [], []]
        program = input[0]
        state_lens = input[2]
        for line_num in range(length):
            statement = program[:state_lens[line_num]]
            program = program[state_lens[line_num]:]
            # obtain Q-value according to target network of actor, state(observation)=current_lower_state
            predicted = actor.get_target_output(current_lower_state, vector[line_num], scope="target")
            states.append([current_lower_state, vector[line_num]])
            action = np.argmax(predicted.cpu()).item()
            actions.append(action)

            if action == 1:  # action{1: retain, 0: delete}
                # current_lower_state , computed by getNextHiddenState function
                out_d, current_lower_state = critic.forward_lstm(current_lower_state, vector[line_num], scope="target")
                # if action==0, store lines (statements, consisting of many tokens) in input_clean
                input_clean[0].extend(statement)
                input_clean[2].append(state_lens[line_num])

            if sum(actions) < 2 and line_num == length - 3 and action != 1:
                input_clean[0].extend(statement)
                input_clean[2].append(state_lens[line_num])
                actions[-1] = 1
        return input_clean

    def __getitem__(self, item):
        x = self.originInputs[item]
        y = self.labels[item]
        length = len(x[2])
        with torch.no_grad():
            x_clean = self.Sampling_RL(self.actorModel, self.criticModel, x,
                                       self.criticModel.proglines_encode(x), length)
        return x_clean, y

    def __len__(self):
        return len(self.labels)

    def get_reTrainData(self):
        new_dataset = pd.DataFrame(columns=['token_indexs_locations', 'label'])
        for x, y in zip(self.originInputs, self.labels):
            length = len(x[2])
            max_len = min(length, self.config.max_sen_length)  # 行数最大不能超过max_sen_length
            with torch.no_grad():
                x_clean = self.Sampling_RL(self.actorModel, self.criticModel, x,
                                           self.criticModel.proglines_encode(x), max_len)
            dict = {'token_indexs_locations': x_clean, 'label': y}
            new_dataset = new_dataset.append(pd.Series(dict), ignore_index=True)
        return new_dataset


if __name__ == '__main__':
    td = RinputData()
    td.append([1, 2, 3, 4, 5], [9, 8, 7, 6, 6])
    # loader = DataLoader(td, shuffle=True, batch_size=3, num_workers=3)
    # loader.sampler.data_source.append([6,7,8])
    import random

    indexs = random.sample(range(0, len(td)), 3)
    results = [td[index][0] for index in indexs]
    train, target = results[:][0], results[:][1]
    print(train, target)
