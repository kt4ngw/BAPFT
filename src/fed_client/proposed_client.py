from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
criterion = F.cross_entropy
mse_loss = nn.MSELoss()

from .base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler

class Proposed_CLient(Client):
    def __init__(self, options, id, model, optimizer, local_dataset, system_attr,  ):

        super(Proposed_CLient, self).__init__(options, id, model, optimizer, local_dataset, system_attr, )

    def local_train(self, round_i):
        begin_time = time.time()
        # 得到原型！！！
        # 得到模型参数
        # print("更新前", self.get_flat_model_params())
        local_model_paras, dict, local_model_protos = self.local_update(self.local_dataset, self.options,  round_i)
        # local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
        # local_model_protos = self.get_local_proto(self.local_dataset, self.options, )
        # print("更新后", self.get_flat_model_params())
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats, (self.local_data_class_distribution, local_model_protos)
    
    def local_update(self, local_dataset, options,  round_i):
        local_seed = 2025 + round_i  # 例如根据 round_i 保证每一轮不同，但可控
        # 创建本地随机生成器
        g = torch.Generator()
        g.manual_seed(local_seed)

        if options['batch_size'] == -1:
            localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True, generator=g)
        else:
            if len(local_dataset) < options['batch_size']:
                localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True, generator=g)
            else:

                sampler = RandomSampler(local_dataset, replacement=False, num_samples=options['batch_size'], generator=g)
                # indices = list(sampler)
                # print("被抽取的样本编号:", indices)
                localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)

        self.model.train()
        train_loss = train_acc = train_total = 0
        last_epoch_proto = {}  # 用于保存最后一个 epoch 的 prototype

        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                self.optimizer.zero_grad()
                feature, pred = self.model(X)  # 取出 feature 和 pred
                loss = criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
                # 👉 在最后一个 epoch 中保存 feature（聚合到 prototype）
                if epoch == options['local_epoch'] - 1:
                    protos = feature.clone().detach()
                    for i in range(len(y)):
                        label = y[i].item()
                        if label not in last_epoch_proto:
                            last_epoch_proto[label] = []
                        last_epoch_proto[label].append(protos[i])
                    local_protos = get_protos(last_epoch_proto) 

        local_model_paras = self.get_flat_model_params()
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return local_model_paras, return_dict, local_protos

    # def local_update(self, local_dataset, options, ):
    #     # print("更新前", self.get_flat_model_params())
    #     # batch_size=options['batch_size']
    #     if options['batch_size'] == -1:
    #         localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
    #     else:
    #         if len(local_dataset) < options['batch_size']:
    #             localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=True)
    #         else:
    #             sampler = RandomSampler(local_dataset, replacement=False, num_samples=1 * options['batch_size'])
    #             # indices = list(sampler)
    #             # print("被抽取的样本编号:", indices)
    #             localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], sampler=sampler)
    #             # localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
    #     self.model.train()
    #     train_loss = train_acc = train_total = 0
    #     for epoch in range(options['local_epoch']):
    #         train_loss = train_acc = train_total = 0
    #         for X, y in localTrainDataLoader:
    #             if self.gpu >= 0:
    #                 X, y = X.cuda(), y.cuda()
    #             self.optimizer.zero_grad()
    #             _, pred = self.model(X)
    #             loss = criterion(pred, y)
    #             loss.backward()
    #             self.optimizer.step()

    #             _, predicted = torch.max(pred, 1)
    #             correct = predicted.eq(y).sum().item()
    #             target_size = y.size(0)
    #             train_loss += loss.item() * y.size(0)
    #             train_acc += correct
    #             train_total += target_size
    #     local_model_paras = self.get_flat_model_params()
    #     return_dict = {"id": self.id,
    #                    "loss": train_loss / train_total,
    #                    "acc": train_acc / train_total}
    #     # print("更新后", self.get_flat_model_params())
    #     return local_model_paras, return_dict
     
    # def get_local_proto(self, local_dataset, options):
    #     # 保存当前状态
    #     was_training = self.model.training  # 保存 train()/eval() 状态
    #     cpu_rng_state = torch.get_rng_state()
    #     cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    #     localTrainDataLoader = DataLoader(local_dataset, batch_size=len(local_dataset), shuffle=False)

    #     local_protos_list = {}
        
    #     self.model.eval()  # 切换到 eval 模式，避免 BN 更新

    #     with torch.no_grad():
    #         for X, y in localTrainDataLoader:
    #             if self.gpu >= 0:
    #                 X, y = X.cuda(), y.cuda()
    #             feature, pred = self.model(X)
    #             protos = feature.clone().detach()
    #             for i in range(len(y)):
    #                 label = y[i].item()
    #                 if label not in local_protos_list:
    #                     local_protos_list[label] = []
    #                 local_protos_list[label].append(protos[i])
        
    #     local_protos = get_protos(local_protos_list)





        # 恢复模型原状态
        if was_training:
            self.model.train()
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)

        return local_protos

def get_protos(protos):
    protos_mean = {}
    for [label, proto_list] in protos.items():
        proto = 0 * proto_list[0]
        for i in proto_list:
            proto += i
        protos_mean[label] = proto / len(proto_list)

    return protos_mean

