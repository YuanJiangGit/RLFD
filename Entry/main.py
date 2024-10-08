import os
import pickle
import sys
from datetime import datetime

import torch
from runnet import RunNet
from critic import critic
from Config.ConfigT import MyConf
from actor import actor
import pandas as pd
from gensim.models.word2vec import Word2Vec
import time
import numpy as np
from collections import Counter
from transformers import RobertaTokenizer


def actorTrain():
    config.IoU = 0

    print('The IoU of testing set is %s' % config.IoU)
    print('Actor Training!')

    bias = 1

    for epoch in range(5-bias+1):
        start_time = time.time()  # 开始时间
        if epoch != 3252:
            rnet.train_actor(criticModel, actorModel, train_data, test_data, epoch+bias, RL_train=True, is_batch_save=False)
        train_end_time = time.time()  # train结束时间
        print("eval!!!")
        # evaluate
        IoU_f, acc_f, recall_f, precision_f, f1_f, top5, top10, top20, MFR, LOC = rnet.eval_actor(criticModel, actorModel, test_data, 99, 138)
        test_end_time = time.time()  # test结束时间
        print('Training Time Cost: %.3f s, Testing Time Cost: %.3f s' % (
            train_end_time - start_time, test_end_time - train_end_time))

        print(f"epoch:{epoch+bias}\nIoU:{IoU_f}\nAcc:{acc_f}\nrecall:{recall_f}\nprecision:{precision_f}\nf1:{f1_f}\ntop5_recall:{top5}\ntop10_recall:{top10}")

        if IoU_f >= config.IoU:
            config.IoU = IoU_f

        torch.save(actorModel.state_dict(), config.models_path + f'/RFLD.pt')
        print("----Model Saved-----")

    print("Reinforcement Done!!!!")

def TrainProcess():
    actorTrain()


def eval_critic():

    test_result_iSeVCs, test_result_program, test_loss = rnet.eval_model(criticModel, test_data, scope='critic')
    test_a, test_p, test_r, test_f = test_result_iSeVCs
    test_a_, test_p_, test_r_, test_f_, FPR = test_result_program
    print(
        '  Test for iSeVCs: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
            test_a, test_p, test_r, test_f))
    print(
        '   Test for programs: [Acc: %.3f, FPR :%.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
            test_a_, FPR, test_p_, test_r_, test_f_))


def eval_join():

    actorModel.load_state_dict(
        torch.load(config.models_path + '%s-%s_actor_with_delay_joint_new.pt' % (config.samplecnt, config.Test_set)))

    test_result, test_loss, IoU = rnet.eval_model_RL(criticModel, actorModel, train_data)
    test_a, test_p, test_r, test_f = test_result
    print(
        'Test.Loss: {%s}, Test: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], Test.IoU: {%.3f}' % (
            test_loss, test_a, test_p, test_r, test_f, IoU))

def eval_actor(flag, model_path, good, bad):
    actorModel.load_state_dict(torch.load(model_path))
    if flag == 1:
        IoU_f, acc_f, recall_f, precision_f, f1_f, top5, top10, top20, MFR, LOC = rnet.eval_actor(criticModel, actorModel, test_data, good, bad)
        print(f"IoU:{IoU_f}\ntop5%_acc:{top5}\ntop10%_acc:{top10}")

    else:
        good, bad = rnet.eval_not_actor(criticModel,actorModel, test_data)
        print(f"good:{good}\nbad:{bad}")


def set_seed():
    np.random.seed(123456)
    torch.manual_seed(123456)

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    data, program_ids, fine_labels, labels, funcs = [], [], [], [], []
    # indexes=[] # jy
    for _, item in tmp.iterrows():
        data.append(item['token_indexs_locations'])
        fine_labels.append(item['label'])
        program_ids.append(item['program_id'])
        funcs.append(item['orig_code'])
        # indexes.append(item) # jy
        if item['label'] == [0]:
            labels.append(0)
        else:
            labels.append(1)
        # labels.append(item['label'])
    return data, torch.LongTensor(labels), fine_labels, program_ids, funcs


class TokenFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 input_ids,
                 row_idx):
        self.input_ids = input_ids
        self.row_idx = row_idx


def convert_examples_to_features(func, label):
    tokenizer = RobertaTokenizer.from_pretrained("./resources/codebert-base")
    block_size = 512
    seg_num = 4
    args_block_size = 512
    args_seg_num = 4

    # source
    rows = str(func).split('\n')
    rows = ['\n' if x == '' else x for x in rows]
    code_tokens = [tokenizer.tokenize(x) for x in rows if tokenizer.tokenize(x) != []]
    row_idx = [[idx + 1] * len(row_token) for idx, row_token in enumerate(code_tokens)]

    code_tokens = [y for x in code_tokens for y in x]  # 平铺token_list
    row_idx = [y for x in row_idx for y in x]

    res = [[], [], []]
    res[0].extend(code_tokens)

    row_token_num = Counter(row_idx)

    res[2].extend(list(row_token_num.values()))

    if label == [0]:
        for i in range(len(res[0])):
            res[1].append(1)

    else:
        for i in row_idx:
            if i in label:
                res[1].append(1)
            else:
                res[1].append(0)

    return res




if __name__ == '__main__':
    set_seed()
    config = MyConf('./Config/config.cfg')
    train_data = pd.read_pickle(config.data_path + 'target/train.pkl')
    print(f"train_data size : {len(train_data)}")

    test_data = pd.read_pickle(config.data_path + 'target/TP_test.pkl')

    print(f"test_data size : {len(test_data)}")

    word2vec = Word2Vec.load(config.embedding_path + "/node_w2v_60").wv
    config.embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    config.embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors
    config.embedding_dim = word2vec.vectors.shape[1]
    config.vocab_size = word2vec.vectors.shape[0] + 1

    rnet = RunNet(config)
    criticModel = critic()
    actorModel = actor(config)

    criticModel.cuda()
    actorModel.cuda()

    # TrainProcess()
    print("******************eval_begin*******************")
    eval_actor(1, "./resources/SavedModels/RFLD.pt", 99, 138)
