from src.fed_server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam       
import copy
class FedAvgTrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, cpu_frequency, B, transmit_power ):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) # , weight_decay=0.001
        super(FedAvgTrainer, self).__init__(options, dataset, clients_label, cpu_frequency, B, transmit_power, model, self.optimizer,)
    
    def train(self):
        print('=== Select {} clients per round ===\n'.format(round((1 - self.dropout_rate) * self.clients_num)))
        #print("第一轮的模型", self.latest_global_model['fc2.bias'])
        for round_i in range(self.num_round):

            self.test_latest_model_on_testdata(round_i)
            bandwidth_allocation_result = self.bandwidth_allocation.equal_allocation(self.clients)
            latency_cost, waiting_time = self.cost.get_latency_sum(self.clients, bandwidth_allocation_result, round_i)


            local_latency = latency_cost[1]
            upload_latency = latency_cost[2]            
            self.cost.accumulated_latency += latency_cost[0]
            self.metrics.update_cost(round_i, local_latency, upload_latency, self.cost.accumulated_latency)    
            self.metrics.update_waiting_time(round_i, waiting_time)
            # 因为计算速度，我们先假设dropout一部分客户端，再训练，而不是训练后dropout      
            no_dropout_clients = self.dropout_clients(round_i=round_i)
            D = self.get_each_class_vloume(no_dropout_clients)
            local_model_paras_set, stats = self.local_train(round_i, no_dropout_clients)    
            print(D)
            # print(local_model_paras_set)
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            print("聚合后", self.latest_global_model)
            # self.optimizer.adjust_learning_rate(round_i)
            self.optimizer.soft_decay_learning_rate()
        self.test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    def dropout_clients(self, round_i=0):
        num_clients = round((1 - self.dropout_rate) * self.clients_num)
        local_rng = np.random.RandomState(seed=self.options['seed'] + round_i)
        index = local_rng.choice(len(self.clients), num_clients, replace=False)
        no_dropout_clients = [self.clients[i] for i in index]
        return no_dropout_clients
