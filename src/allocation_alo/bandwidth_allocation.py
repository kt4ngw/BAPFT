from src.utils.tool_utils import setup_seed

import numpy as np
setup_seed(2024)
class Bandwidth_Allocation():
    def __init__(self, options, total_bandwidth):
        self.options = options
        self.total_bandwidth = total_bandwidth
        self.latency_upper = 1000
        self.latency_lower = 0
        self.V = 0

    # def allocation_bandwidth(self, total_bandwidth, selected_clients, V):
    #     result = [0 for i in range(len(selected_clients))]
    #     latency_A = self.latency_upper
    #     while(V = 0):
    #         for i in range(len(selected_clients)):
    #             result[i] = 0 # [7]中的公式（8）计算得出。
    #         allocated_bandwidth = sum(result)
    #         if allocated_bandwidth >  self.total_bandwidth:
    #             latency_A = (latency_A + self.latency_upper) / 2
    #             self.latency_lower = latency_A
    #         else if allocated_bandwidth < alpha * B:
    #             V = 1
    #             spare_bandwidth = total_bandwidth - allocated_bandwidth
    #             # ---- # 补充其他分配
    #             new_result = self.energy_bandwidth_allocate(result, spare_bandwidth)
    #         else:
    #             latency_A = (latency_A + self.latency_lower) / 2
    #             self.latency_upper = latency_A
    #     return new_result 
    # def energy_bandwidth_allocate(self, result, spare_bandwidth, selected_clients):
        
    #     new_result = [0 for i in range(len(selected_clients))]
    #     for i in range(len(selected_clients)):
    #         new_result[i] = result[i] + () * spare_bandwidth

    #     return new_result 


    def equal_allocation(self, selected_clients):
        bandwitdh_allocation_result = [1 / len(selected_clients) \
                                        * self.total_bandwidth for i in range(len(selected_clients))]

        return bandwitdh_allocation_result


    def proposed_bandwidth_allocation(self, selected_clients, round_i, baseline2021=False):
        latency_upper = 10000
        latency_lower = max([selected_clients[i].getLocalDelay(round_i) for i in range(len(selected_clients))])
        result = [0 for i in range(len(selected_clients))]
        latency_A = latency_upper
        V = 0 
        while(V == 0):
            for i in range(len(selected_clients)):
                result[i] = self.proposed_ba_comp_allcaotion(selected_clients[i], latency_A, round_i)
            allocated_bandwidth = sum(result)
            if baseline2021 == True:
               self.options["weight"] = 1 
            if allocated_bandwidth <  ((self.options["weight"] * (1 - 0.6)  + 0.6) * self.total_bandwidth) and allocated_bandwidth > ((self.options["weight"] * (1 - 0.6)  + 0.6) - 0.01) * self.total_bandwidth:
                V = 1
            else:
                if allocated_bandwidth > (self.options["weight"] * (1 - 0.6)  + 0.6) * self.total_bandwidth:
                    latency_lower = latency_A
                    latency_A = (latency_A + latency_upper) / 2

                else:
                    latency_upper = latency_A   
                    latency_A = (latency_A + latency_lower) / 2
        spare_bandwidth = self.total_bandwidth - allocated_bandwidth 
        new_result = self.energy_bandwidth_allocate(result, spare_bandwidth, selected_clients)    
        return new_result   

    def energy_bandwidth_allocate(self, result, spare_bandwidth, selected_clients):
        # allocation = [result[i] / sum(result) for i in range(len(result))]
        temp = 0
        for i in range(len(selected_clients)):
            # t = (result[i] * np.log2(1 + selected_clients[i].attr_dict['transmit_power'] * 8)) ** 2
            # temp += 1 / t
            t = (self.options['model_size'] * selected_clients[i].attr_dict['transmit_power']) / (result[i] * self.total_bandwidth)  ** 2 * np.log2(1 + selected_clients[i].attr_dict['transmit_power'] * 8)
            temp += t
        allocation = [((self.options['model_size'] * selected_clients[i].attr_dict['transmit_power']) / (result[i] * self.total_bandwidth)  ** 2 * np.log2(1 + selected_clients[i].attr_dict['transmit_power'] * 8)) / temp for i in range(len(result))]
        #print("allocation", allocation)
        # 
        #print("result", result)               
        new_result = [0 for i in range(len(selected_clients))]
        for i in range(len(selected_clients)):
          #  print("spare_bandwidth", spare_bandwidth)
            #print((allocation[i]) * spare_bandwidth)
            new_result[i] = result[i] + (allocation[i]) * spare_bandwidth
        return new_result 

    def proposed_ba_comp_allcaotion(self, client, latency_A, round_i):
        from scipy.special import lambertw
        B_min_MHz = 1 * 1e-9
        B_max_MHz = 99
        import math
        T_avail = float(latency_A - client.getLocalDelay(round_i))
        # print("latency_A", latency_A)
        # if T_avail <= 0:
        #     return np.inf
        Pt_dBm = float(client.attr_dict['transmit_power'])  # in [0,23] dBm
        Pt_W = 10.0 ** ((Pt_dBm - 30.0) / 10.0)
        h_lin = 10.0 ** (-9.73)  # channel gain in linear scale
        N0_WHz = 10.0 ** ((-174.0 - 30.0) / 10.0)
        a = (Pt_W * float(h_lin)) / float(N0_WHz)  

        C = (float(self.options['model_size'] * 8 * 1024 * 1024) / float(T_avail)) * math.log(2.0)
        k = C / a       
        if not (0.0 < k < 1.0):
            return np.inf
        
        z0 = -k * math.exp(-k)
        Wm1 = lambertw(z0, k=-1).real
        denom = Wm1 + k
        if denom >= 0.0:                     # 保险判断
            return np.inf

        B_hz = - C / denom
        if not np.isfinite(B_hz) or B_hz <= 0.0:
            return np.inf
        B_hz = float(B_hz)

        B_hz = B_hz * 1e-6
        # print("B_hz", B_hz)
        # print("B_mhz", B_hz)
        # 区间裁剪策略：小于下限取下限；大于上限记为不可达
        # if B_hz < 0:
        #     return B_hz
        # if B_hz > B_max_MHz:
        #     return np.inf
        # print("upload", float(self.options['model_size'] * 8 * 1024 * 1024) / (B_hz * 1000000 * np.log2(1 + (10**((client.attr_dict['transmit_power'] -30) /10) * 10**(-9.73)) / (B_hz * 1000000 * (10 ** (-20.4))))))
        # print(T_avail)
        return B_hz




    def baseline2021_bandwidth_allocation(self, selected_clients, round_i, baseline2021=True):
        latency_upper = 10000
        latency_lower = max([selected_clients[i].getLocalDelay(round_i) for i in range(len(selected_clients))])
        result = [0 for i in range(len(selected_clients))]
        latency_A = (latency_upper  + latency_lower) / 2
        V = 0 
        while(V == 0):
            for i in range(len(selected_clients)):
                result[i] = self.proposed_ba_comp_allcaotion(selected_clients[i], latency_A, round_i)
            allocated_bandwidth = sum(result)
            # if baseline2021 == True:
            #    self.options["weight"] = 1 
            if allocated_bandwidth <  (1 * self.total_bandwidth) and allocated_bandwidth > (1 - 0.01) * self.total_bandwidth:
                V = 1
            else:
                if allocated_bandwidth > 1 * self.total_bandwidth: 
                    latency_lower = latency_A  
                    latency_A = (latency_A + latency_upper) / 2     
                else:
                    latency_upper = latency_A 
                    latency_A = (latency_A + latency_lower) / 2 
        # print(latency_A)

        return result  

    def jcsba(self, selected_clients, selected_index_latency):
        print(selected_index_latency)
        result = [selected_index_latency[i] / sum(selected_index_latency) * self.total_bandwidth for i in range(len(selected_clients))]
        print(result)
        return result


    def jacsba_one_allocation(self, selected_clients):
        result = [1.0 * self.total_bandwidth]
        return result
        
    def random_allocation(self, selected_clients):
        """
        随机分配总带宽给选定的客户端，每个客户端最多分配不超过2%的总带宽。
        分配总量保持不变。
        """
        num_clients = len(selected_clients)
        max_share = 0.02 * self.total_bandwidth  # 每个客户端的最大带宽

        # 初始随机权重
        random_weights = np.random.rand(num_clients)
        normalized_weights = random_weights / np.sum(random_weights)
        initial_allocations = [w * self.total_bandwidth for w in normalized_weights]

        # 限制在最大允许带宽范围内
        allocations = [min(alloc, max_share) for alloc in initial_allocations]
        total_allocated = sum(allocations)

        # 计算剩余带宽并分配给未达到上限的客户端
        remaining_bandwidth = self.total_bandwidth - total_allocated
        under_limit_indices = [i for i, alloc in enumerate(allocations) if alloc < max_share]

        if under_limit_indices and remaining_bandwidth > 0:
            additional_weights = np.random.rand(len(under_limit_indices))
            additional_weights /= np.sum(additional_weights)
            for i, weight in zip(under_limit_indices, additional_weights):
                extra = min(max_share - allocations[i], remaining_bandwidth * weight)
                allocations[i] += extra

        return allocations