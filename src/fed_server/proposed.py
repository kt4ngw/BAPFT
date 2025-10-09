from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import copy
from src.fed_client.proposed_client import Proposed_CLient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import random
class Proposed(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(Proposed, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model, self.optimizer,)
        self.global_proto = None

    def train(self):
        print('=== Select {} clients per round ===\n'.format(round((1 - self.dropout_rate) * self.clients_num)))
        #print("第一轮的模型", self.latest_global_model['fc2.bias'])
        pre_finetune_model = copy.deepcopy(self.latest_global_model)
        lr = 0.1
        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)
            
            self.latest_global_model = pre_finetune_model
            # self.test_latest_model_on_testdata(round_i)

            bandwidth_allocation_result = self.bandwidth_allocation.baseline2021_bandwidth_allocation(self.clients, round_i)
            print(bandwidth_allocation_result)
            # print("bandwidth_allocation_result", sum(bandwidth_allocation_result))
            latency_cost, waiting_time = self.cost.get_latency_sum(self.clients, bandwidth_allocation_result, round_i)
            print("waiting_time", waiting_time)
            local_latency = latency_cost[1]
            upload_latency = latency_cost[2]   
            self.cost.accumulated_latency += latency_cost[0]
            # 计算每个客户端的等待时间~
            self.metrics.update_cost(round_i, local_latency, upload_latency, self.cost.accumulated_latency)    
            self.metrics.update_waiting_time(round_i, waiting_time)
            # 因为计算速度，我们先假设dropout一部分客户端，再训练，而不是训练后dropout    
            no_dropout_clients = self.dropout_clients(round_i=round_i)
            D = self.get_each_class_vloume(no_dropout_clients)
            # print(D)
            local_model_paras_set, stats, local_model_proto_set = self.local_train(round_i, no_dropout_clients)  

            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            self.global_proto = self.aggregate_protos(local_model_proto_set)
            # print("聚合后", self.latest_global_model)
            pre_finetune_model = copy.deepcopy(self.latest_global_model)
            # ↓↓↓ 加入原型微调步骤 ↓↓↓
            # if round_i % 50 == 0 and round_i != 0:
            self.optimizer.soft_decay_learning_rate()
            all_local_protos = [proto_dict for _, proto_dict in local_model_proto_set]
            X_mixup, y_mixup = self.generate_mixup_features_from_all_clients(
                all_local_protos, self.global_proto, num_per_class=20, alpha_range=(0.2, 0.5))
            
            # lr *= 0.99
            # if round_i > 50:
            #     lr = lr // 2
            self.prototype_mixup_classifier_finetune(X_mixup.cuda(), y_mixup.cuda(), epochs=5, lr=0.05)

            # if round_i > 50 and round_i != 0:
            # proto_tensor = self.prepare_global_proto_tensor(self.global_proto)
            # self.prototype_only_classifier_finetune(proto_tensor, epochs=5, lr=0.01)
            self.latest_global_model = self.get_flat_model_params()
            if round_i in [50, 100]:
                visualize_mixup_distribution(X_mixup, all_local_protos, self.global_proto, round_i)

            # print("微调后", self.latest_global_model)
            
        self.test_latest_model_on_testdata(self.num_round)

        self.metrics.write()

    def generate_mixup_features_from_all_clients(self, all_local_protos, global_proto, num_per_class, alpha_range):
        from collections import defaultdict
        import torch
        import random

        class_to_local_protos = defaultdict(list)
        for local_proto_dict in all_local_protos:
            for label, local_proto in local_proto_dict.items():
                class_to_local_protos[label].append(local_proto)

        mixup_features = []
        mixup_labels = []

        for label, local_proto_list in class_to_local_protos.items():
            if label not in global_proto:
                continue
            global_p = global_proto[label]
            for _ in range(num_per_class):
                # 直接使用全局 random
                local_proto = random.choice(local_proto_list)
                # 直接使用全局 torch 随机数
                alpha = torch.empty(1).uniform_(*alpha_range).item()
                mixed = alpha * local_proto + (1 - alpha) * global_p
                mixup_features.append(mixed.unsqueeze(0))
                mixup_labels.append(torch.tensor([label]))

        return torch.cat(mixup_features), torch.cat(mixup_labels)



    def prepare_global_proto_tensor(self, global_proto_dict):
        sorted_keys = sorted(global_proto_dict.keys())
        global_proto_tensor = torch.stack([global_proto_dict[k] for k in sorted_keys]).cuda()
        return global_proto_tensor  # [C, D]

    def prototype_mixup_classifier_finetune(self, X, y, epochs, lr):
        self.set_flat_model_params(self.latest_global_model)
        # optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            logits = self.model.classifier(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def dropout_clients(self, round_i=0):
        num_clients = round((1 - self.dropout_rate) * self.clients_num)
        local_rng = np.random.RandomState(seed=self.options['seed'] + round_i)
        index = local_rng.choice(len(self.clients), num_clients, replace=False)
        no_dropout_clients = [self.clients[i] for i in index]
        return no_dropout_clients

    def local_train(self, round_i, select_clients):
        local_model_paras_set = []
        stats = []
        local_model_proto_set = []
        for i, client in enumerate(select_clients, start=1):
            client.set_flat_model_params(self.latest_global_model)
            local_model_paras, stat, local_model_proto = client.local_train(round_i)
            local_model_proto_set.append(local_model_proto)
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            # if True:
            #     print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
            #           "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
            #            round_i, client.id, i, round((1 - self.dropout_rate) * self.clients_num),
            #            stat['loss'], stat['acc']*100, stat['time'], ))
        return local_model_paras_set, stats, local_model_proto_set
    
    def setup_clients(self, dataset, clients_label):
        train_data = dataset.trainData
        train_label = dataset.trainLabel
        all_client = []
        for i in range(len(clients_label)):
            local_client = Proposed_CLient(self.options, i, self.model, self.optimizer, TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.clients_system_attr)
            all_client.append(local_client)

        return all_client
    

     
    def aggregate_parameters(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_global_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        self.simple_average = False
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            num = 0
            for num_sample, local_solution in solns:
                # print(local_solution)
                num += num_sample
                averaged_solution += num_sample * local_solution
            averaged_solution /= num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()
    
    def aggregate_protos(self, protos_set, **kwargs):
        agg_protos_label = {}
        agg_sizes_label = {}    
        for local_sizes, local_protos in protos_set:
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                    agg_sizes_label[label].append(local_sizes[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
                    agg_sizes_label[label] = [local_sizes[label]]
        for [label, protos_list] in agg_protos_label.items():
            sizes_list = agg_sizes_label[label]
            proto = 0 * protos_list[0]
            for i in range(len(protos_list)):
                proto += sizes_list[i] * protos_list[i]
            agg_protos_label[label] = proto / sum(sizes_list)
        return agg_protos_label
    
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io

def convert_npz_to_mat(npz_file, mat_file):
    """
    把 .npz 文件转换为 .mat 文件，便于 MATLAB 使用。
    """
    data = np.load(npz_file)
    scipy.io.savemat(mat_file, {
        'X_pca': data['X_pca'],
        'labels': data['labels']
    })
    print(f'[Info] Saved .mat file to {mat_file}')

def visualize_mixup_distribution(X_mixup, all_local_protos, global_proto, round_i, save_dir='visualization'):
    """
    可视化混合原型、各客户端本地原型、全局原型在二维 PCA 空间中的分布，并保存到文件。
    额外保存 PCA 投影数据为 .npz 和 .mat 文件，便于 MATLAB 加载。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 拼接所有本地原型
    local_proto_list = []
    for proto_dict in all_local_protos:
        local_proto_list.extend(proto_dict.values())
    local_proto_tensor = torch.stack(local_proto_list).cpu()

    # 全局原型 tensor
    global_proto_tensor = torch.stack(list(global_proto.values())).cpu()

    # 拼接全部特征
    X_all = torch.cat([
        X_mixup.cpu(),
        local_proto_tensor,
        global_proto_tensor
    ], dim=0).numpy()

    # 来源标签：0 - mixup, 1 - local, 2 - global
    labels = (
        [0] * X_mixup.size(0) +
        [1] * local_proto_tensor.size(0) +
        [2] * global_proto_tensor.size(0)
    )
    labels = np.array(labels)

    # PCA 降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    # 保存 PCA 投影数据 (.npz)
    save_data_path = os.path.join(save_dir, f'round_{round_i:03d}_pca_data.npz')
    np.savez(save_data_path, X_pca=X_pca, labels=labels)
    print(f'[Info] Saved PCA data to {save_data_path}')

    # 转存为 .mat 文件
    save_mat_path = os.path.join(save_dir, f'round_{round_i:03d}_pca_data.mat')
    convert_npz_to_mat(save_data_path, save_mat_path)

    # 绘制可视化图
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:X_mixup.size(0), 0], X_pca[:X_mixup.size(0), 1], c='r', label='Mixup', alpha=0.5)
    plt.scatter(X_pca[X_mixup.size(0): X_mixup.size(0) + local_proto_tensor.size(0), 0],
                X_pca[X_mixup.size(0): X_mixup.size(0) + local_proto_tensor.size(0), 1],
                c='g', label='Local Protos', alpha=0.5)
    plt.scatter(X_pca[-global_proto_tensor.size(0):, 0], X_pca[-global_proto_tensor.size(0):, 1],
                c='b', label='Global Proto', s=60, edgecolor='k')
    plt.legend()
    plt.title(f'Round {round_i}: PCA of Mixup / Local / Global Protos')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)

    # 保存图片
    save_img_path = os.path.join(save_dir, f'round_{round_i:03d}_mixup_distribution.png')
    plt.savefig(save_img_path)
    plt.close()
    print(f'[Info] PCA visualization saved to {save_img_path}')