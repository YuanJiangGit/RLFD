import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_curve
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import threading
from sklearn.utils import shuffle
import time
from queue import Queue
import random
from collections import deque, Counter
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

from Models.StagedModel_long_seq_p import Model
from utils import RinputData, ReTrainCritic


class TokenFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 row_idx):
        self.input_ids = input_ids
        self.row_idx = row_idx


class RunNet:
    def __init__(self, config):
        self.config = config
        self.loss_fn = F.cross_entropy
        self.prob_fn = F.softmax
        self.reTrainCritic = False
        self.eps = 1.0
        self.x_info = {'token_index': 0, 'val_location': 1, 'lines_len': 2}

    def prob_np_fn(self, x):
        '''
        计算抽样的概率
        :param x:
        :return:
        '''
        prob_array = np.exp(x) / np.sum(np.exp(x), axis=0)
        return prob_array

    def clip_gradient(self, model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def Sampling_RL(self, actor, critic, input, vector, length):
        '''
        :param actor:
        :param critic:
        :param inputs: a sentence (size: 1* 句子长度)
        :param vector: the embedding (1 * length*embedding matrix) corresponding to the sentence
        :param length: the length of the sentence
        :param epsilon: the epsilon of epsilon greedy
        :param Random: consider random exploration
        :return:
        '''
        num_layers = self.config.lstm_num_layers
        num_directions = 2
        # current_lower_state = torch.zeros(1, num_layers * num_directions * self.config.lstm_hidden_dim).cuda()
        actions = []
        states = []
        # new input derived by actor and the former of new input is the same as the input
        input_clean = [[], [], []]
        word_input_clean = []
        program = input[0]
        state_lens = input[2]

        for line_num in range(length):
            statement = program[:state_lens[line_num]]
            program = program[state_lens[line_num]:]
            # obtain Q-value according to target network of actor, state(observation)=current_lower_state
            predicted, _ = actor(None, vector[line_num], scope="target")
            states.append([None, vector[line_num]])
            action = np.argmax(predicted.cpu()).item()
            actions.append(action)

            if action == 1:  # action{1: retain, 0: delete}
                # current_lower_state , computed by getNextHiddenState function
                # out_d, current_lower_state = critic.forward_lstm(current_lower_state, vector[line_num],
                #                                                  scope="target")
                # if action==0, store lines (statements, consisting of many tokens) in input_clean
                input_clean[0].extend(statement)
                input_clean[2].append(state_lens[line_num])

            if sum(actions) < 2 and line_num == length - 3 and action != 1:
                input_clean[0].extend(statement)
                input_clean[2].append(state_lens[line_num])
                actions[-1] = 1

        return actions, states, input_clean

    # 处理多线程的episode采样
    def Sampling_RL_Thread(self, actor, critic, input, vector, label_f, length, epsilon, y, q, Random=True):
        actions = []
        states = []
        preds = []
        nc_preds = []
        # new input derived by actor and the former of new input is the same as the input
        input_clean = [[], [], []]
        program = input[0]
        state_lens = input[2]
        vector = vector.unsqueeze(0)
        label_f = label_f.unsqueeze(0)
        nc_logits, logits, _, my_actions = actor(None, vector, labels_f=label_f, scope="target")

        for line_num in range(length):
            statement = program[:state_lens[line_num]]
            program = program[state_lens[line_num]:]
            predicted = logits[0][line_num]
            nc_predicted = logits[0][line_num]
            states.append([None, vector[0][line_num]])
            preds.append(predicted)
            nc_preds.append(nc_predicted)
            if Random:
                if random.random() < self.eps:
                    if random.random() < 0.5:
                        action = 0
                    else:
                        action = 1
                else:
                    if predicted[0] >= predicted[1]:
                        action = 0
                    else:
                        action = 1

                # b = predicted[1].item()
                # a = 1-b

                # action = np.random.choice(np.array(range(2)), p=[a, b])
            else:
                if predicted[0] >= predicted[1]:
                    action = 0
                else:
                    action = 1
            actions.append(action)  # 1*1

            if action == 1:  # action{1: retain, 0: delete}这里的retain指的是作为漏洞语句集保留
                input_clean[0].extend(statement)
                input_clean[2].append(state_lens[line_num])

            # 至少保留一行？
            if sum(actions) < 2 and line_num == length - 3 and action != 1:
                input_clean[0].extend(statement)
                input_clean[2].append(state_lens[line_num])
                actions[-1] = 1

        # 拼接Rinput和Oinput，
        Input2critic = [input, input]
        # tokens_features, _, _ = self.get_critic_input(Input2critic, [0 , 0], [[0], [0]])
        # tokens_features = [[y.cuda() for y in x] for x in tokens_features]
        # critic.train(False)
        # out = critic(tokens_features, scope="target")
        # probs = self.prob_fn(out, dim=1)[0][y.item()]  # 返回每个类的概率
        # probs = self.prob_fn(out, dim=0)[:, y.item()]  # 返回正确类别所对应的概率  (input_clean为尽量去除无关代码的漏洞语句集合)
        probs = torch.tensor([[1], [1]])
        # for i in range(len(actions)):
        #     actions[i] = my_actions[0][i]
        q.put({'actions': actions, 'status': states, 'Input2critic': Input2critic, 'probs': probs, 'preds': preds,
               'nc_preds': nc_preds})

    def grad_RL_Thread(self, actions, states, R, actorModel, L, label_f, preds):
        # R:IoU-aveIoU 越高越好
        # print(len(actions))
        for pos in range(len(actions)):
            rr = [0, 0]
            # 给rr中的第0位或者第1位赋值， 如果第0位被赋值，那么就采取的动作0， 否则采取的动作1（动作x的奖励）
            # rr[actions[pos]] = (R * self.config.alpha).cpu().item()
            rr[actions[pos]] = R.cpu().item()
            # statelist[i][pos][0] 第pos个statement的隐藏层输出（1,2hidden_size）,2*hidden_size由hidden和cell组成
            # statelist[i][pos][1] 第pos个statement的向量
            grad_current = actorModel.get_gradient(states[pos][0], states[pos][1], rr, label_f[pos], preds[pos],
                                                   scope="target")
            L.append(grad_current)
        # print(2)

    def cac_reward(self, prob_, Input2critic, actions, y_fine, predslist):
        # prob_update = self.cac_reward(prob_, Input2critic, sample_result['actions'], y_fine)

        pred_vul_line = [index + 1 for (index, value) in enumerate(actions) if value == 1]
        # 预测对的句子个数
        intersec_nums = len(set(pred_vul_line).intersection(y_fine))
        union_nums = len(set(pred_vul_line).union(y_fine))

        # 原始句子的长度
        L_total = len(Input2critic[1][2])
        # 删除的句子的长度
        L_delete = float(L_total - len(Input2critic[0][2]))
        # 损失需要加上number(delete)/number(the total number of words) ** 2
        # 更新Rinput的reward值，用于后续调整actor的参数  +( L_delete/ L_total) * 0.1
        # prob_update = prob_[0] + (prob_[0] - prob_[1]) * 0.2 + (L_delete / L_total) * 0.2
        if self.reTrainCritic:
            prob_update = prob_[0] - prob_[1] + (intersec_nums / union_nums) * 0.8  # TD at here!
        else:
            prob_update = torch.tensor(intersec_nums / union_nums)

        answer = torch.zeros(len(actions))
        for i in y_fine:
            if i < answer.size(0):
                answer[i] = 1
        trues = np.array(answer)
        preds = np.array(actions)
        acc_f = accuracy_score(preds, trues)

        p = predslist[0]
        line_and_pre = []
        for b in range(len(p)):
            line_and_pre.append((b, p[b][1]))
        sorted_line_and_pre = sorted(line_and_pre, key=lambda x: x[1], reverse=True)

        vul_pos = []
        sample = sorted_line_and_pre
        for i in range(len(sample)):
            if sample[i][0] in y_fine:
                vul_pos.append(i + 1)

        pos_reward = 0
        for i in vul_pos:
            pos_reward += 1 / i
        return torch.tensor(2*pos_reward)

    def train_model(self, criticModel, actorModel, train_data, test_data, epoch, RL_train=True):
        criticModel.cuda()
        actorModel.cuda()
        critic_target_optimizer = torch.optim.Adam(criticModel.target_pred.parameters(), lr=2e-5, eps=1e-8)
        critic_active_optimizer = torch.optim.Adam(criticModel.active_pred.parameters(), lr=2e-5, eps=1e-8)

        actor_target_optimizer = torch.optim.Adam(actorModel.target_policy.parameters(), lr=2e-5, eps=1e-8)
        actor_active_optimizer = torch.optim.Adam(actorModel.active_policy.parameters(), lr=2e-5,
                                                  eps=1e-8)  # ---jy 更新梯度
        steps = 0
        rTC = ReTrainCritic(self.config)
        i, idx = 0, 0
        trues = []
        predicts = []
        avgloss = 0
        # train_data = shuffle(train_data)
        batch_total_num = len(train_data) / self.config.batch_size

        while i < len(train_data):
            batch_aveprob = 0.
            batch = self.get_batch(train_data, i, self.config.batch_size)
            i += self.config.batch_size
            idx += 1
            train_inputs, train_labels, train_fine_labels, _ = batch
            tokens_features, _, _ = self.get_critic_input(train_inputs, train_labels, train_fine_labels)

            if self.config.use_gpu:
                train_inputs, train_labels = train_inputs, train_labels.cuda()
                tokens_features = [[y.cuda() for y in x] for x in tokens_features]

            criticModel.train(False)
            tokens_encodes, _ = criticModel.proglines_encode(tokens_features)

            LSTM_train = self.reTrainCritic
            # print(idx)
            # if i % (30 * 16) == 0:
            #     print(i, "/", len(train_data))
            # if idx > 300:
            #     break

            criticModel.assign_active_network()
            actorModel.assign_active_network()
            # RL_train=False： 根据这一个批次内的数据，一个一个实例的对critic network进行更新（首先对target跟新，再对active更新）
            # RL_train=True, 根据这一个批次内的数据，拿出一个实例，随机抽取samplecnt次trajectory，对每次samplecnt中的每个action都对target进行更新，
            # 将所有samplecnt次trajectory累计得到的梯度再对active进行更新。所以他俩只是梯度更新的方式不同
            for j in range(len(train_inputs)):
                aveprob = 0.
                x = train_inputs[j]
                y = train_labels[j]
                tokens_encode = tokens_encodes[j]
                # there is no need to apply RF in non-vul iSeVCs, since the reward calculated by IoU is 0 which have no effect on parameter adjustment
                if y.cpu().numpy().tolist() == 0:
                    continue
                y_fine = train_fine_labels[j]
                # there is no need to train the policy using the instances whose vulnerable lines are large than max_sen_length
                if y_fine[0] > min(self.config.max_sen_length, tokens_encode.size(0)):
                    continue
                # 将x和y添加到缓存区
                rTC.append(x, y_fine)

                length = len(x[self.x_info['lines_len']])  # 句子的最大长度是config.sentence_len, 不足最大长度补0，计算非0的个数
                max_len = min(length, self.config.max_sen_length, tokens_encode.size(0))  # 行数最大不能超过max_sen_length

                if RL_train:
                    # print("RL True")
                    criticModel.train(
                        False)  # when train(True), it does dropout; when train(False) it doesn’t do dropout
                    actorModel.train()
                    actionlist, statelist, problist, Rinputlist = [], [], [], []
                    aveLoss = 0.
                    # samplecnt 决定对一个句子，采样的次数（相当于samplecnt trajectory）

                    # add mulitiple threading
                    # t_thread_before = time.time()
                    threads = []
                    q = Queue()
                    for _ in range(self.config.samplecnt):
                        t = threading.Thread(target=self.Sampling_RL_Thread,
                                             args=(actorModel, criticModel, x, tokens_encode, max_len,
                                                   self.config.epsilon, y, q))
                        t.start()
                        threads.append(t)
                    [t.join() for t in threads]
                    for _ in range(self.config.samplecnt):
                        sample_result = q.get()
                        # [actions, states, loss_]
                        actionlist.append(sample_result['actions'])
                        statelist.append(sample_result['status'])
                        Input2critic = sample_result['Input2critic']
                        prob_ = sample_result['probs']
                        # 计算采样得到的Rinput的reward，把符合条件的reward放入到Rinputlist，并计算采样的概率
                        prob_update = self.cac_reward(prob_, Input2critic, sample_result['actions'],
                                                      y_fine)  # TD loss or IoU 越高越好
                        aveprob += prob_update.item()
                        problist.append(prob_update)
                    '''
                    if (steps) % 50 == 0:
                        print("-------------------------------------------")
                    '''
                    aveprob /= self.config.samplecnt

                    # 加入多线程的梯度求解
                    # t_thread2_before = time.time()
                    threads = []
                    # q = Queue()
                    L = []
                    for s_th in range(self.config.samplecnt):
                        R = problist[s_th] - aveprob
                        t = threading.Thread(target=self.grad_RL_Thread,
                                             args=(actionlist[s_th], statelist[s_th], R, actorModel, L))
                        t.start()
                        threads.append(t)
                    [t.join() for t in threads]
                    # List中的所有行的第0列对应维度元素相加，依次类推第一列，第二列...
                    # grad_temps = [np.sum([L[m][n] for m in range(len(L))], axis=0) for n in range(len(L[0]))]

                    grad_temps = []
                    for n in range(len(L[0])):
                        a = torch.zeros_like(L[0][n])
                        for m in range(len(L)):
                            a += L[m][n]
                        grad_temps.append(a)

                    # grad_temps=list(np.sum(grad_matrix,axis=0))
                    # print('使用多线程求梯度需要花的时间：%s' % (time.time() - t_thread2_before))
                    actor_target_optimizer.zero_grad()
                    # 自动更新梯度
                    actor_active_optimizer.zero_grad()
                    # actorModel.active_policy.zero_grad()  # 使用模型清零梯度
                    # print("previous grad: ", actorModel.active_policy.b.grad)
                    actorModel.assign_active_network_gradients(grad_temps)
                    # 更新参数
                    actor_active_optimizer.step()
                    batch_aveprob += aveprob
                    # 手动更新梯度
                    # for param in actorModel.active_policy.parameters():
                    #     param.data += (1e-3) * param.grad

                    # 第一次是每隔200批次微调critic，但是之后批次应该减小
                    if LSTM_train and idx % self.config.retrain_critic_sep_batch == 0:
                        self.config.retrain_critic_sep_batch = 30
                        # print("RL and LSTM True")
                        criticModel.train()
                        actorModel.train(False)
                        rTC._update_model(actorModel, criticModel)
                        # prepare data
                        reTrain_data = rTC.get_reTrainData()
                        # 微调critic
                        train_result, train_loss = self.train_model_without_delay(criticModel, reTrain_data, 0)
                        train_a, train_p, train_r, train_f = train_result
                        # 清空数据以及模型
                        rTC._clear()
                        LSTM_train = False

                        print(
                            'Epoch: {%s},Batches: {%s}, Train. Loss: {%s}, Train: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f] ' % (
                                epoch + 1, idx, train_loss, train_a, train_p, train_r, train_f))

                    # print("after: active: ", actorModel.active_policy.b, "target: ", actorModel.target_policy.b)
                else:
                    # print("RL False LSTM True")
                    criticModel.train()
                    actorModel.train(False)
                    critic_active_optimizer.zero_grad()
                    critic_target_optimizer.zero_grad()
                    prediction = criticModel([tokens_features[j], tokens_features[j]], scope="target")[0].unsqueeze(0)
                    trues.append(y.item())  # 用item 可能有问题
                    predicts.append(torch.max(prediction, 1)[1].item())
                    # 存储预测结果
                    # predlist.append(prediction)
                    # targetlist.append(y)
                    loss = self.loss_fn(prediction, y)  # 根据预测来计算损失
                    avgloss += loss.item()
                    loss.backward()  # 根据损失，计算模型参数的梯度
                    criticModel.assign_active_network_gradients()  ## 根据critic 中target 的梯度来更新active的梯度,然后target梯度清零
                    critic_active_optimizer.step()  # 使用optimizer根据梯度对模型的参数进行自动更新（因为target没有梯度所以只更新active）
            # 一个batch结束
            print(
                f"train epoch {epoch} : {i / self.config.batch_size} / {batch_total_num} batch done! batch IoU:{batch_aveprob / self.config.batch_size}")
            if RL_train:
                # print("Again RL True")
                criticModel.train(False)
                actorModel.train()
                # print(actorModel.target_policy.b.data, actorModel.active_policy.b.data)
                actorModel.update_target_network()
                # print(actorModel.target_policy.b.data, actorModel.active_policy.b.data)
                if LSTM_train:
                    # print("Again RL AND LSTM True")
                    criticModel.train()
                    actorModel.train()
                    # print(criticModel.active_pred.label.bias, criticModel.target_pred.label.bias)
                    criticModel.update_target_network()
                    # print(criticModel.active_pred.label.bias, criticModel.target_pred.label.bias)

            else:
                # print("Again RL False and LSTM True")
                criticModel.train()
                actorModel.train(False)
                criticModel.assign_target_network()  # 一个批次完成之后，再更新target

            # 每-轮进行验证, 100 * 16 30 * 16
            if False and idx % (1000000000 * 16) == 0:
                test_result, test_loss, IoU = self.eval_model_RL(criticModel, actorModel, test_data)
                test_a, test_p, test_r, test_f = test_result
                print(
                    'Epoch: {%s}, Batches: {%s}, Test.Loss: {%s}, Test: [Acc: %.3f, prec: %.3f, recall: %.3f, f1 :%.3f], Test.IoU: {%.3f}' % (
                        epoch + 1, idx, test_loss, test_a, test_p, test_r, test_f, IoU))
                if IoU > self.config.IoU:
                    #  '/%s_actor_with_delay_joint.pt'%self.config.Test_set '/%s_critic_with_delay_joint.pt'%self.config.Test_set
                    # torch.save(actorModel.state_dict(), self.config.models_path + '/%s-%s_actor_with_delay_joint.pt'%(self.config.samplecnt,self.config.Test_set))
                    # torch.save(criticModel.state_dict(), self.config.models_path + '/%s-%s_critic_with_delay_joint.pt'%(self.config.samplecnt,self.config.Test_set))
                    # self.config.IoU = IoU

                    torch.save(actorModel.state_dict(),
                               self.config.models_path + '/%s-%s_actor_with_delay_joint_new.pt' % (
                                   self.config.samplecnt, self.config.Test_set))
                    # torch.save(criticModel.state_dict(),
                    #            config.models_path + '/%s-%s_critic_with_delay_joint.pt' % (config.samplecnt, config.Test_set))
                    self.config.IoU = IoU
                    print("----Mdoel Saved-----")
        # 一个epoch结束
        if not RL_train:
            train_a, train_p, train_r, train_f = self.evaluate_b(trues, predicts)
            train_result = (train_a, train_p, train_r, train_f)
            train_loss = avgloss / len(train_data)
            return train_result, train_loss

    def train_actor(self, criticModel, actorModel, train_data, test_data, epoch, RL_train, is_batch_save):
        criticModel.cuda()
        actorModel.cuda()
        actor_target_optimizer = torch.optim.Adam(actorModel.target_policy.parameters(), lr=2e-7, eps=1e-8)
        actor_active_optimizer = torch.optim.Adam(actorModel.active_policy.parameters(), lr=2e-7, eps=1e-8)
        i, idx = 0, 0

        batch_total_num = len(train_data) / self.config.batch_size
        true_locats = []
        pred_locats = []
        while i < len(train_data):
            batch_aveprob = 0.
            batch = self.get_batch(train_data, i, self.config.batch_size)
            i += self.config.batch_size
            idx += 1
            train_inputs, train_labels, train_fine_labels, _ = batch
            tokens_features, label_cs, label_fs = self.get_critic_input(train_inputs, train_labels, train_fine_labels)

            if self.config.use_gpu:
                train_inputs, train_labels = train_inputs, train_labels.cuda()
                tokens_features = [[y.cuda() for y in x] for x in tokens_features]
                (label_cs, label_fs) = [x.cuda() for x in [label_cs, label_fs]]

            criticModel.train(False)
            tokens_encodes = criticModel.proglines_encode(tokens_features)
            actorModel.assign_active_network()
            # actorModel.update_target_network()
            if RL_train == True:
                for j in range(len(train_inputs)):
                    aveprob = 0.
                    x = train_inputs[j]
                    y = train_labels[j]
                    tokens_encode = tokens_encodes[j]
                    label_f = label_fs[j]
                    y_fine = []
                    for g in range(len(label_f)):
                        if label_f[g] == 1:
                            y_fine.append(g)
                    if y_fine == []:
                        continue
                    length = len(x[self.x_info['lines_len']])  # 句子的最大长度是config.sentence_len, 不足最大长度补0，计算非0的个数
                    max_len = min(length, self.config.max_sen_length, tokens_encode.size(0))  # 行数最大不能超过max_sen_length

                    criticModel.train(False)  # when train(True), it does dropout; when train(False) it doesn’t do dropout
                    actorModel.train()
                    actionlist, statelist, problist, Rinputlist, predslist, nc_predslist = [], [], [], [], [], []
                    threads = []
                    q = Queue()
                    for _ in range(self.config.samplecnt):
                        t = threading.Thread(target=self.Sampling_RL_Thread,
                                             args=(actorModel, criticModel, x, tokens_encode, label_f, max_len,
                                                   self.config.epsilon, y, q))
                        t.start()
                        threads.append(t)
                    [t.join() for t in threads]

                    for _ in range(self.config.samplecnt):
                        sample_result = q.get()
                        # [actions, states, loss_]
                        actionlist.append(sample_result['actions'])
                        statelist.append(sample_result['status'])
                        Input2critic = sample_result['Input2critic']
                        prob_ = sample_result['probs']
                        predslist.append(sample_result['preds'])
                        nc_predslist.append(sample_result['nc_preds'])
                        # 计算采样得到的Rinput的reward，把符合条件的reward放入到Rinputlist，并计算采样的概率
                        prob_update = self.cac_reward(prob_, Input2critic, sample_result['actions'],
                                                      y_fine, nc_predslist)  # TD loss or IoU 越高越好
                        aveprob += prob_update.item()
                        problist.append(prob_update)
                    aveprob /= self.config.samplecnt

                    # 加入多线程的梯度求解
                    threads = []
                    L = []
                    for s_th in range(self.config.samplecnt):
                        R = problist[s_th] - aveprob
                        t = threading.Thread(target=self.grad_RL_Thread,
                                             args=(actionlist[s_th], statelist[s_th], R, actorModel, L, label_f,
                                                   predslist[s_th]))
                        t.start()
                        threads.append(t)
                    [t.join() for t in threads]
                    # List中的所有行的第0列对应维度元素相加，依次类推第一列，第二列...

                    grad_temps = []
                    for n in range(len(L[0])):
                        if L[0][n] is None:
                            grad_temps.append(None)
                        else:
                            a = torch.zeros_like(L[0][n])
                            for m in range(len(L)):
                                a += L[m][n]
                            grad_temps.append(a)
                    # 自动更新梯度
                    actor_target_optimizer.zero_grad()
                    actor_active_optimizer.zero_grad()
                    actorModel.assign_active_network_gradients(grad_temps)
                    actor_active_optimizer.step()
                    batch_aveprob += aveprob
            else:
                # a = criticModel(tokens_features, 'target', label_cs, label_fs)
                actorModel.train(True)
                logits_f, labels_f = actorModel(None, tokens_encodes, scope="target", labels_c=label_cs,
                                                labels_f=label_fs)  # batchsize*linenum*2*1
                true_locat = []
                pred_locat = []
                for a in range(logits_f.size(0)):
                    logit_f = logits_f[a]
                    label_f = labels_f[a]
                    for b in range(logits_f.size(1)):
                        if label_f[b].item() == -1:
                            break
                        if label_f[b].item() == 0:
                            true_locat.append(0)
                        else:
                            true_locat.append(1)

                        if logit_f[b][0].item() >= logit_f[b][1].item():
                            pred_locat.append(0)
                        else:
                            pred_locat.append(1)

                true_locats.append(true_locat)
                pred_locats.append(pred_locat)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                logits_f = logits_f.reshape(-1, 2)
                labels_f = labels_f.reshape(-1)
                loss_f = loss_fct(logits_f, labels_f.long())
                loss_f.backward()
                actor_target_optimizer.step()
                actor_target_optimizer.zero_grad()
                batch_aveprob += loss_f
            # 一个batch结束
            # 只算为漏洞的程序的IoU
            print(
                f"train epoch {epoch} : {i / self.config.batch_size} / {batch_total_num} batch done! batch IoU(or loss):{batch_aveprob / self.config.batch_size}")
            if RL_train:
                actorModel.update_target_network()
            else:
                pass


    def train_critic(self, criticModel, actorModel, train_data, test_data, epoch, RL_train):
        criticModel.cuda()
        critic_target_optimizer = torch.optim.Adam(criticModel.target_policy.parameters(), lr=2e-5, eps=1e-8)
        i, idx = 0, 0

        batch_total_num = len(train_data) / self.config.batch_size
        while i < len(train_data):
            batch_aveprob = 0.
            batch = self.get_batch(train_data, i, self.config.batch_size)
            i += self.config.batch_size
            idx += 1
            train_inputs, train_labels, train_fine_labels, _ = batch
            tokens_features, label_cs, label_fs = self.get_critic_input(train_inputs, train_labels, train_fine_labels)

            if self.config.use_gpu:
                tokens_features = [[y.cuda() for y in x] for x in tokens_features]
                (label_cs, label_fs) = [x.cuda() for x in [label_cs, label_fs]]

            criticModel.train(True)
            loss_f = criticModel(None, tokens_features, scope="target", labels_c=label_cs, labels_f=label_fs)
            loss_f.backward()
            critic_target_optimizer.step()
            critic_target_optimizer.zero_grad()
            batch_aveprob += loss_f
            # 一个batch结束
            print(
                f"train epoch {epoch} : {i / self.config.batch_size} / {batch_total_num} batch done! batch IoU(or loss):{batch_aveprob / self.config.batch_size}")

    def train_model_without_delay(self, model, train_data, optimizer):
        model.train()
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        # train_data = shuffle(train_data)
        while i < len(train_data):
            batch = self.get_batch(train_data, i, self.config.batch_size)
            i += self.config.batch_size
            train_inputs, train_labels, train_fine_labels, _ = batch
            tokens_features, _, _ = self.get_critic_input(train_inputs, train_labels, train_fine_labels)

            if self.config.use_gpu:
                train_inputs, train_labels = train_inputs, train_labels.cuda()
                tokens_features = [[y.cuda() for y in x] for x in tokens_features]

            model.zero_grad()
            output = model(tokens_features, scope="target")

            loss = self.loss_fn(output, train_labels)
            loss.backward()
            optimizer.step()

            trues.extend(train_labels.cpu().numpy().tolist())
            predicts.extend(torch.max(output, 1)[1].cpu().data.numpy().tolist())
            # calc training acc
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)
        train_a, train_p, train_r, train_f = self.evaluate_b(trues, predicts)
        train_result = (train_a, train_p, train_r, train_f)
        train_loss = total_loss / total
        return train_result, train_loss

    def eval_model(self, model, test_data, scope):
        predicts = []
        trues = []
        program_ids = []
        # indexes_lst=[] # jy
        total_loss = 0.0
        total = 0.0
        i = 0

        batch_total_num = len(test_data) / self.config.batch_size
        logits_c = []

        # model.train(False)
        model.eval()
        with torch.no_grad():
            while i < len(test_data):
                batch = self.get_batch(test_data, i, self.config.batch_size)
                i += self.config.batch_size
                # test_inputs, test_labels, _, test_program_ids = batch
                test_inputs, test_labels, test_fine_labels, test_program_ids = batch  # jy

                tokens_features, _, _ = self.get_critic_input(test_inputs, test_labels, test_fine_labels)
                if self.config.use_gpu:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                    tokens_features = [[y.cuda() for y in x] for x in tokens_features]

                model.target_pred.batch_size = len(test_labels)
                if scope == 'critic':
                    output = model(tokens_features, scope="target")
                    logits_c.append(output)

                if scope == 'actor':
                    output = model(test_inputs, scope="target")

                # if self.config.use_gpu:
                #     test_labels = torch.tensor([test_labels]).cuda()
                loss = self.loss_fn(output, Variable(test_labels))
                # loss = self.loss_fn(output, test_labels)
                # calc valing acc
                # _, predicted = torch.max(output.data, 1)
                # predicts.extend(torch.max(output, 1)[1].cpu().data.numpy().tolist())
                prob = torch.softmax(output, dim=-1)
                for tmp in range(output.size(0)):
                    if output[tmp][0] > output[tmp][1]:
                        predicts.append(0)
                    else:
                        predicts.append(1)

                trues.extend(test_labels.cpu().numpy().tolist())
                program_ids.extend(test_program_ids)
                # indexes_lst.extend(indexes) # jy
                total += len(test_labels)
                total_loss += loss.item() * len(test_inputs)
                print(f"eval batch {i / self.config.batch_size} / {batch_total_num} done!")

        a = np.array(predicts)
        b = np.array(trues)
        f1_c = f1_score(b, a)
        print("*********************")
        print(f1_c)
        print("*********************")

        aaa = np.sum(a)

        # df = pd.DataFrame[{'pre_rl':a, 'true_rl':b}]
        # df.to_csv('pre_rl.csv',index=False)
        test_loss = total_loss / total
        test_a, test_p, test_r, test_f = self.evaluate_b(trues, predicts)
        test_a_, test_p_, test_r_, test_f_, FPR = self.evaluate_binary(trues, predicts, program_ids)
        test_result_iSeVCs = (test_a, test_p, test_r, test_f)
        test_result_program = (test_a_, test_p_, test_r_, test_f_, FPR)
        return test_result_iSeVCs, test_result_program, test_loss

    def eval_model_RL(self, criticModel, actorModel, test_data):
        predicts = []
        trues = []
        pred_locats = []
        true_locats = []
        total_loss = 0.0
        total = 0.0
        i = 0
        # test_data = shuffle(test_data)
        batch_total_num = len(test_data) / self.config.batch_size
        with torch.no_grad():
            idx = 0
            while i < len(test_data):
                batch = self.get_batch(test_data, i, self.config.batch_size)
                i += self.config.batch_size
                test_inputs, test_labels, test_fine_labels, _ = batch
                tokens_features, _, _ = self.get_critic_input(test_inputs, test_labels, test_fine_labels)

                # !!!
                # if i/self.config.batch_size < 23:
                #     continue

                if self.config.use_gpu:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                    tokens_features = [[y.cuda() for y in x] for x in tokens_features]

                criticModel.train(False)
                tokens_encodes, _ = criticModel.proglines_encode(tokens_features)
                idx += 1
                # if idx % 500 == 0:
                #     print(idx, "/", len(test_data))
                # if idx > 200:
                #     break
                for j in range(len(test_labels)):
                    # if i/self.config.batch_size > 9:
                    #     print(1)
                    x = test_inputs[j]
                    y = test_labels[j]
                    y_fine = test_fine_labels[j]
                    tokens_encode = tokens_encodes[j]
                    length = len(x[2])
                    max_len = min(length, self.config.max_sen_length, tokens_encode.size(0))  # 行数最大不能超过max_sen_length
                    actions, states, input_clean = self.Sampling_RL(actorModel, criticModel, x,
                                                                    tokens_encode, max_len)
                    # determine input for critic
                    input = input_clean if self.reTrainCritic else x

                    new_tokens_features, _, _ = self.get_critic_input([input, input], [y, y], [y_fine, y_fine])
                    # print(x, Rinput, length, Rlenth)
                    # if (i % 50) == 0:
                    # print(actions)
                    new_tokens_features = [[y.cuda() for y in x] for x in new_tokens_features]
                    output = criticModel(new_tokens_features, scope="target")[0].unsqueeze(0)

                    if self.config.use_gpu:
                        y = torch.tensor([y]).cuda()
                    loss = self.loss_fn(output, y)
                    # 预测最大值的索引（0或1）
                    predicts.append(torch.max(output, 1)[1].cpu().data.numpy().tolist())
                    trues.append(y.cpu().numpy().tolist())

                    pred_vul_line = [index + 1 for (index, value) in enumerate(actions) if value == 1]
                    pred_locats.append(pred_vul_line)
                    true_locats.append(y_fine)

                    total += 1
                    total_loss += loss.item()
                print(f"eval batch {i / self.config.batch_size} / {batch_total_num} done!")

        test_loss = total_loss / total
        # test_a, test_p, test_r, test_f = self.evaluate_b(trues, predicts)
        test_a, test_p, test_r, test_f = self.evaluate_b([x[0] for x in trues], [x[0] for x in predicts])
        test_result = (test_a, test_p, test_r, test_f)
        # IoU = self.cal_IoU(true_locats, pred_locats)
        # 只算预测为漏洞的程序的IoU
        IoU = self.cal_IoU_real(true_locats, pred_locats, predicts)
        return test_result, test_loss, IoU
        # return total_epoch_loss / 200, total_epoch_acc / 200

    def get_pred(self, criticModel, actorModel, test_data):
        trues = []
        pred_locats = []
        true_locats = []
        y_trues_fs = []
        y_preds_fs = []
        line_and_pres = []
        i = 0
        # test_data = shuffle(test_data)
        batch_total_num = len(test_data) / self.config.batch_size
        with torch.no_grad():
            idx = 0
            while i < len(test_data):
                batch = self.get_batch(test_data, i, self.config.batch_size)
                i += self.config.batch_size
                test_inputs, test_labels, test_fine_labels, _ = batch
                tokens_features, label_cs, label_fs = self.get_critic_input(test_inputs, test_labels, test_fine_labels)

                if self.config.use_gpu:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                    tokens_features = [[y.cuda() for y in x] for x in tokens_features]
                    (label_cs, label_fs) = [x.cuda() for x in [label_cs, label_fs]]

                criticModel.train(False)
                tokens_encodes = criticModel.proglines_encode(tokens_features)
                idx += 1
                actorModel.train(False)
                logits_f, _ , labels_f, _ = actorModel(None, tokens_encodes, scope="target", labels_c=label_cs,
                                                   labels_f=label_fs)
                # logits_f, labels_f = criticModel(tokens_features, "target", label_cs, label_fs)
                for a in range(logits_f.size(0)):
                    logit_f = logits_f[a]
                    pred_locat = []
                    for b in range(logits_f.size(1)):
                        if logit_f[b][0] < logit_f[b][1]:
                            pred_locat.append(b)
                    pred_locats.append(pred_locat)
        return pred_locats[0]

    def eval_actor(self, criticModel, actorModel, test_data, good_num, bad_num):
        trues = []
        miss_num = 0
        pred_locats = []
        true_locats = []
        y_trues_fs = []
        y_preds_fs = []
        line_and_pres = []
        i = 0

        batch_total_num = len(test_data) / self.config.batch_size
        test_orig_code = test_data["orig_code"].tolist()
        with torch.no_grad():
            idx = 0
            while i < len(test_data):
                batch = self.get_batch(test_data, i, self.config.batch_size)
                batch_orig_code = test_orig_code[i:i+self.config.batch_size]
                i += self.config.batch_size
                test_inputs, test_labels, test_fine_labels, _ = batch
                tokens_features, label_cs, label_fs = self.get_critic_input(test_inputs, test_labels, test_fine_labels)

                if self.config.use_gpu:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                    tokens_features = [[y.cuda() for y in x] for x in tokens_features]
                    (label_cs, label_fs) = [x.cuda() for x in [label_cs, label_fs]]

                criticModel.train(False)
                tokens_encodes = criticModel.proglines_encode(tokens_features)
                idx += 1
                actorModel.train(False)
                logits_f, _, labels_f, _ = actorModel(None, tokens_encodes, scope="target", labels_c=label_cs, labels_f=label_fs)

                for a in range(logits_f.size(0)):
                    logit_f = logits_f[a]
                    label_f = labels_f[a]
                    true_locat = []
                    pred_locat = []
                    for b in range(logits_f.size(1)):
                        if label_f[b].item() == -1:
                            break

                        if label_f[b].item() == 1:
                            true_locat.append(b)

                        if logit_f[b][0] < logit_f[b][1]:
                            pred_locat.append(b)
                    if true_locat == []:
                        miss_num += 1
                        continue
                    true_locats.append(true_locat)
                    pred_locats.append(pred_locat)
                # print(f"eval batch {i/self.config.batch_size} / {batch_total_num} done!")

                for a in range(logits_f.size(0)):
                    logit_f = logits_f[a]
                    label_f = labels_f[a]
                    y_trues_f = []
                    y_preds_f = []
                    line_and_pre = []
                    for b in range(logits_f.size(1)):
                        if label_f[b].item() == -1:
                            break

                        if label_f[b].item() == 1:
                            y_trues_f.append(1)
                        else:
                            y_trues_f.append(0)

                        if logit_f[b][0] < logit_f[b][1]:
                            y_preds_f.append(1)
                        else:
                            y_preds_f.append(0)

                        line_and_pre.append((b, logit_f[b][1]))
                    if np.array(y_trues_f).sum() == 0:
                        continue
                    y_trues_fs.append(y_trues_f)
                    y_preds_fs.append(y_preds_f)

                    line_and_pres.append(sorted(line_and_pre, key=lambda x: x[1], reverse=True))
                print(f"eval batch {i / self.config.batch_size} / {batch_total_num} done!")
        ks = [1, 3, 5, 10]
        topk_recall = []
        for k in ks:
            sum = 0
            for l in range(len(line_and_pres)):
                sample = line_and_pres[l][:k]
                flag = 0
                for i in sample:
                    if i[0] in true_locats[l]:
                        flag = 1
                sum += flag
            topk_recall.append(sum / len(line_and_pres))

        # per_ks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.05, 0.1]
        per_ks = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.05]
        per_topk_recall = []
        t5s = []
        t10s = []
        for k in per_ks:
            sum = 0
            for l in range(len(line_and_pres)):
                n = len(line_and_pres[l])
                sample = line_and_pres[l][:math.ceil(n * k)]
                for i in line_and_pres[l]:
                    if abs(i[1] - sample[0][1]) < 0.001:
                        aaa = 1
                        sample.append(i)
                flag = 0
                for i in sample:
                    if i[0] in true_locats[l]:
                        flag = 1

                if flag == 1:
                    if k == 0.05:
                        t5s.append(1)
                    if k == 0.1:
                        t10s.append(1)
                else:
                    if k == 0.05:
                        t5s.append(0)
                    if k == 0.1:
                        t10s.append(0)
                sum += flag
            per_topk_recall.append(sum / (len(line_and_pres) + bad_num + good_num + miss_num))

        MFRs = []
        LOCs = []
        for l in range(len(line_and_pres)):
            sample = line_and_pres[l]
            sum = 0
            max_sum_pos = 0
            for i in range(len(sample)):
                if sample[i][0] in true_locats[l]:
                    sum += 1
                    max_sum_pos = i
                    if sum == 1:
                        MFRs.append(i)
            LOCs.append(max_sum_pos)

        LOC = 0
        MFR = 0
        for i in LOCs:
            LOC += i
        LOC = LOC / len(LOCs)
        for i in MFRs:
            MFR += i
        MFR = MFR / len(MFRs)

        IoU_fs = []
        my_trues = []
        my_preds = []
        for i in range(len(y_preds_fs)):
            my_trues.extend(y_trues_fs[i])
            my_preds.extend(y_preds_fs[i])
            y_trues_f = np.array(y_trues_fs[i])
            y_preds_f = np.array(y_preds_fs[i])
            IoU_fs.append((y_preds_f & y_trues_f.astype(int)).sum() / (y_preds_f | y_trues_f.astype(int)).sum())

        a = pred_locats
        b = true_locats
        c = IoU_fs
        d = t5s
        e = t10s
        f = line_and_pres

        all_res = [a, b, c, d, e, f]
        all_df = pd.DataFrame(all_res).T
        # all_df.to_excel("find_exp_119.xlsx", index=False)

        for i in range(miss_num):
            IoU_fs.append(0.0)

        IoU_f = np.mean(IoU_fs)

        my_trues = np.array(my_trues)
        my_preds = np.array(my_preds)
        acc_f = accuracy_score(my_trues, my_preds)
        recall_f = recall_score(my_trues, my_preds)
        precision_f = precision_score(my_trues, my_preds)
        f1_f = f1_score(my_trues, my_preds)
        fpr, _, _ = roc_curve(my_trues, my_preds)
        print("fpr")
        print(fpr)
        print("fpr")

        return IoU_f, acc_f, recall_f, precision_f, f1_f, per_topk_recall[-1], per_topk_recall[4], per_topk_recall[
            -2], MFR, LOC

    def eval_not_actor(self, criticModel, actorModel, test_data):
        pred_locats = []
        none_num = 10
        i = 0
        batch_total_num = len(test_data) / self.config.batch_size
        with torch.no_grad():
            idx = 0
            while i < len(test_data):
                batch = self.get_batch(test_data, i, self.config.batch_size)
                i += self.config.batch_size
                test_inputs, test_labels, test_fine_labels, _ = batch
                tokens_features, label_cs, label_fs = self.get_critic_input(test_inputs, test_labels, test_fine_labels)

                if self.config.use_gpu:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                    tokens_features = [[y.cuda() for y in x] for x in tokens_features]
                    (label_cs, label_fs) = [x.cuda() for x in [label_cs, label_fs]]

                criticModel.train(False)
                tokens_encodes = criticModel.proglines_encode(tokens_features)
                idx += 1
                actorModel.train(False)
                # logits_f, _, labels_f = actorModel(None, tokens_encodes, scope="target", labels_c=label_cs, labels_f=label_fs)
                logits_f, labels_f = criticModel(tokens_features, "target", label_cs, label_fs)
                for a in range(logits_f.size(0)):
                    logit_f = logits_f[a]
                    pred_locat = []
                    for b in range(logits_f.size(1)):
                        if logit_f[b][0] < logit_f[b][1]:
                            pred_locat.append(b)
                    pred_locats.append(pred_locat)
                print(f"eval batch {i/self.config.batch_size} / {batch_total_num} done!")
        good = 0
        bad = 0
        for i in pred_locats:
            if i == []:
                good += 1
            else:
                bad += 1
        bad = bad - none_num

        return good, bad

    def get_TPFP(self, criticModel, actorModel, test_data):
        TP = []
        FP = []
        test = []
        n = 0
        i = 0
        batch_total_num = len(test_data) / self.config.batch_size
        with torch.no_grad():
            idx = 0
            while i < len(test_data):
                batch = self.get_batch(test_data, i, self.config.batch_size)
                i += self.config.batch_size
                test_inputs, test_labels, test_fine_labels, _ = batch
                tokens_features, label_cs, label_fs = self.get_critic_input(test_inputs, test_labels, test_fine_labels)

                if self.config.use_gpu:
                    test_inputs, test_labels = test_inputs, test_labels.cuda()
                    tokens_features = [[y.cuda() for y in x] for x in tokens_features]
                    (label_cs, label_fs) = [x.cuda() for x in [label_cs, label_fs]]

                criticModel.train(False)
                tokens_encodes = criticModel.proglines_encode(tokens_features)
                idx += 1
                actorModel.train(False)
                # logits_f, _, labels_f = actorModel(None, tokens_encodes, scope="target", labels_c=label_cs, labels_f=label_fs)
                logits_c, labels_f = criticModel(tokens_features, "target", label_cs, label_fs)
                for a in range(logits_c.size(0)):
                    if logits_c[a][0] < logits_c[a][1]:
                        TP.append(n)
                        test.append(1)
                    else:
                        FP.append(n)
                        test.append(0)
                    n += 1
                print(f"eval batch {i/self.config.batch_size} / {batch_total_num} done!")

        return TP, FP

    def evaluate_b(self, y, y_pred):
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, \
            recall_score, \
            f1_score
        import re
        report_text = classification_report(y, y_pred, target_names=['nsbr', 'sbr'])
        # print(report_text)
        report_list = re.sub(r'[\n\s]{1,}', ' ', report_text).strip().split(' ')
        conf_matrix = confusion_matrix(y, y_pred)
        # print(conf_matrix)
        TN = conf_matrix.item((0, 0))
        FN = conf_matrix.item((1, 0))
        TP = conf_matrix.item((1, 1))
        FP = conf_matrix.item((0, 1))
        prec = 100 * precision_score(y, y_pred, average='binary')
        recall = 100 * recall_score(y, y_pred, average='binary')
        f_measure = 100 * f1_score(y, y_pred, average='binary')

        accuracy = 100 * accuracy_score(y, y_pred)
        return accuracy, prec, recall, f_measure

    def evaluate_binary(self, y, y_pred, program_ids):
        # def evaluate_binary(self, y, y_pred, program_ids, indexes_lst):
        # program_y={}
        # for index, program_id in enumerate(program_ids):
        #     if program_id in program_y:
        #         if y[index] != y_pred[index] or y[index]==1:
        #             program_y[program_id]=(y[index],y_pred[index])
        #     else:
        #         program_y[program_id] = (y[index], y_pred[index])
        #
        # y, y_pred = zip(*program_y.values())
        FP_programs = []
        FN_programs = []
        # wrong_indexes=[]
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(y_pred)):
            if y[i] == y_pred[i] == 1:
                TP += 1
            if y_pred[i] == 1 and y[i] != y_pred[i]:
                FP += 1
                FP_programs.append(program_ids[i])
                # wrong_indexes.append(indexes_lst[i])
            if y[i] == y_pred[i] == 0:
                TN += 1
            if y_pred[i] == 0 and y[i] != y_pred[i]:
                FN += 1
                FN_programs.append(program_ids[i])
                # wrong_indexes.append(indexes_lst[i])
        recall = 100 * TP / (TP + FN)
        prec = 100 * TP / (TP + FP)
        FPR = 100 * FP / (FP + TN)
        if 0 == prec + recall:
            f_measure = 0
        else:
            f_measure = 2 * prec * recall / (prec + recall)

        if 0 == TP + TN + FP + FN:
            accuracy = 0
        else:
            accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)

        print('FP programs:')
        print(FP_programs)
        print('FN programs:')
        print(FN_programs)
        # print('Wrong indexes:')
        # print(wrong_indexes)
        return accuracy, prec, recall, f_measure, FPR

    def cal_IoU(self, true_locats, pred_locats):
        pre_lens = []
        iou_list = []
        for true_locat, pred_locat in zip(true_locats, pred_locats):
            if true_locat == [0]:
                continue
            else:
                pre_lens.append(len(pred_locat))
                inter = set(true_locat).intersection(set(pred_locat))
                union = set(true_locat).union(set(pred_locat))
                _iou = len(inter) * 1.0 / len(union)
                iou_list.append(_iou)
        ave_IoU = np.mean(iou_list)
        v = np.mean(pre_lens)
        print('The average number of detected vulnerable lines of source code: %s' % v)
        return float(ave_IoU)

    def cal_IoU_real(self, true_locats, pred_locats, predicts):
        '''
        calculated IoU for real_world programs
        :param true_locats:
        :param pred_locats:
        :return:
        '''
        pre_lens = []
        iou_list = []
        for true_locat, pred_locat, predict in zip(true_locats, pred_locats, predicts):
            pre_lens.append(len(pred_locat))
            inter = set(true_locat).intersection(set(pred_locat))
            union = set(true_locat).union(set(pred_locat))
            _iou = len(inter) * 1.0 / len(union)
            iou_list.append(_iou)
        ave_IoU = np.mean(iou_list)
        v = np.mean(pre_lens)
        print('The average number of detected vulnerable lines of source code: %s' % v)
        return float(ave_IoU)

    def get_batch(self, dataset, idx, bs):
        tmp = dataset.iloc[idx: idx + bs]
        data, program_ids, fine_labels, labels = [], [], [], []
        # indexes=[] # jy
        for _, item in tmp.iterrows():
            data.append(item['token_indexs_locations'])
            fine_labels.append(item['label'])
            program_ids.append(item['program_id'])
            # if item['label'] == [0]:
            #     labels.append(0)
            # else:
            #     labels.append(1)
            labels.append(item['target'])
        return data, torch.LongTensor(labels), fine_labels, program_ids

    ## parameter setting
    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.learning_rate * (0.5 ** (epoch // 6))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def get_critic_input(self, inputs, label_cs, label_fs):
        feature_dataset = self.FeatureDataset(inputs, label_cs, label_fs)
        feature_dataloader = DataLoader(feature_dataset, sampler=SequentialSampler(feature_dataset),
                                        batch_size=len(inputs), num_workers=0)
        feature_iter = iter(feature_dataloader)
        tokens_features, label_cs, label_fs = next(feature_iter)
        return tokens_features, label_cs, label_fs

    class FeatureDataset(Dataset):
        def __init__(self, inputs, label_cs, label_fs):
            self.inputs = inputs
            self.label_cs = label_cs
            self.label_fs = label_fs
            self.tokenizer = RobertaTokenizer.from_pretrained("./resources/codebert-base")
            self.block_size = 512
            self.seg_num = 4

        def __len__(self):
            return len(self.inputs)

        def convert_examples_to_features(self, input):

            code_tokens = input[0]  # 平铺token_list

            row_idx = []
            for i in range(len(input[2])):
                for j in range(input[2][i]):
                    row_idx.append(i + 1)

            tokens_features = []
            # row_num = row_idx[-2]q
            start_idx = 0
            seg_num = math.ceil((len(code_tokens) + 1) / self.block_size)
            seg_num = min(seg_num, self.seg_num)
            row_num = 0

            for i in range(seg_num):
                end_token_idx = start_idx + self.block_size - 2
                if end_token_idx < len(code_tokens) - 1:
                    last_row = row_idx[end_token_idx]
                    end_idx = row_idx.index(last_row)
                else:
                    end_idx = len(code_tokens)

                seq_source_tokens = [self.tokenizer.cls_token] + code_tokens[start_idx:end_idx]
                seq_row_indices = [0] + (np.array(row_idx[start_idx:end_idx]) - row_idx[start_idx] + 1).tolist()
                row_num += seq_row_indices[-1]

                seq_input_ids = self.tokenizer.convert_tokens_to_ids(seq_source_tokens)

                padding_length = self.block_size - len(seq_input_ids)
                seq_input_ids += [self.tokenizer.pad_token_id] * padding_length

                tokens_feature = TokenFeatures(seq_input_ids, seq_row_indices)
                tokens_features.append(tokens_feature)

                start_idx = end_idx

            padding_length = self.seg_num - len(tokens_features)
            tokens_features += [None] * padding_length

            return tokens_features, row_num

        def get_feature(self, feature):
            max_length = self.block_size
            if feature is None:
                return (torch.ones(max_length).long(),
                        torch.zeros(max_length, max_length).bool(),
                        torch.zeros(max_length, max_length),
                        torch.zeros(max_length).bool(),
                        torch.zeros(1).bool())
            token_row_idx = feature.row_idx
            row_token_nums = Counter(token_row_idx)
            row_idx = [np.where(np.array(token_row_idx) == x)[0][0] for x in row_token_nums.keys()] + [
                len(token_row_idx)]
            row_num = len(row_token_nums)

            # self-attention maskz
            attn_mask = torch.zeros(max_length, max_length)
            attn_mask[:len(token_row_idx), :len(token_row_idx)] = 1
            attn_mask = attn_mask.bool()

            # token行坐标
            row2row_mask = torch.zeros(max_length, max_length)
            for idx in range(row_num):
                row2row_mask[row_idx[idx]:row_idx[idx + 1], row_idx[idx]:row_idx[idx + 1]] = 1

            row2row_mask[0, :len(token_row_idx)] = 1

            # 每行第一个token位置
            row_mask = torch.zeros(max_length)
            row_mask[row_idx[:-1]] = 1
            row_mask = row_mask.bool()

            return (torch.tensor(feature.input_ids),
                    attn_mask,
                    row2row_mask,
                    row_mask,
                    torch.ones(1).bool())

        def __getitem__(self, i):
            max_length = self.block_size
            input = self.inputs[i]
            label_c = self.label_cs[i]
            label_f = self.label_fs[i]
            tokens_feature, row_num = self.convert_examples_to_features(input)
            c = label_c.clone().detach()

            tmp = torch.zeros(row_num)
            if isinstance(label_f, list):
                # !!!!!c
                # vul_idx = [x for x in label_f if x < row_num]
                vul_idx = [x+1 for x in label_f if x+1 < row_num]
                try:
                    tmp[vul_idx] = 1
                except Exception as e:
                    print(e)
            tmp = tmp.tolist()
            tmp += [-1] * (max_length - row_num)

            f = torch.tensor(tmp)
            a = [self.get_feature(tokens_feature[i]) for i in range(self.seg_num)]
            return a, c, f
