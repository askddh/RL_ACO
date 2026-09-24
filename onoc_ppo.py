import copy
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from typing import Dict, List, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
import torch.optim as optim
from torch.distributions import Categorical
from . import MR

class RLOpticalNetworkEnv(gym.Env):
    """
    RL环境用于光网络映射和波长分配
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, C, communication):
        super(RLOpticalNetworkEnv, self).__init__()   #确保父类被正确初始化

        self.C = C  # 核心列表 (名字列表)
        self.communication = communication  # 通信关系 dict
        self.ONI = list(range(len(self.C)))  # 光网络节点 (可以使用索引列表)
        self.num_cores = len(C)
        self.num_nodes = len(self.ONI)
        self.link_id_map, self.id_link_map = self._build_link_id_map()
        self.num_links = len(self.link_id_map)

        # 生成通信矩阵W
        self.W = self._generate_W(C, communication)
        self.distance_matrix = self._get_distance_matrix()

        # 波长数量等于通信节点数
        self.num_wavelengths = self.num_nodes

        # 波长参数
        self.λ0 = 1550.0
        self.FSR = 59.0
        self.communication_wavelengths = [self.λ0 + i * (self.FSR / self.num_wavelengths) for i in range(self.num_wavelengths)]
        

        # 计算正确的状态维度
        self.phase2_state_dim = (self.num_links
                                + self.num_wavelengths
                                + self.num_links * self.num_wavelengths
                                )

        self.state_dim = (1
                        + self.num_cores * self.num_nodes
                        + self.phase2_state_dim
                        )


        # RL状态和动作空间
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # 动作空间：选择节点（映射阶段）或选择波长（分配阶段）
        self.action_space = spaces.Discrete(max(self.num_nodes, self.num_wavelengths))

        # 环境状态
        self.state = None
        self.info = None
        self.state, self.info = self.reset()

    def _generate_W(self, C, communication):
        """生成通信权重矩阵"""
        num_cores = len(C)
        W = np.zeros((num_cores, num_cores))
        for src, targets in communication.items():
            if src not in C:
                continue
            src_index = C.index(src)
            for dest, weight in targets:
                if dest not in C:
                    continue
                dest_index = C.index(dest)
                W[src_index][dest_index] = weight
        return W
    
    def _get_distance_matrix(self):
        """计算环形网络距离矩阵（假设环状编号 0..n-1）"""
        n = len(self.ONI)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    distance_matrix[i][j] = j - i
                    distance_matrix[j][i] = n - (j - i)
                elif i == j:
                    distance_matrix[i][j] = 0
        return distance_matrix
    
    def _build_link_id_map(self):
        """
        为所有物理链路分配一个唯一 ID
        ring 网络：i -> (i+1)%N
        """
        link_id = {}
        id_link = {}
        idx = 0
        N = self.num_nodes
        for i in range(N):
            j = (i + 1) % N
            link_id[(i, j)] = idx
            id_link[idx] = (i, j)
            idx += 1
        return link_id, id_link

    def _get_state(self):
        """
        Unified state for Phase 1 (mapping) and Phase 2 (wavelength assignment)
        """
        phase_flag = np.array([self.phase], dtype=np.float32) #标识当前phase
        # 当前要映射任务核，core_mapping_state[i]=1表示为第i个任务核选择节点
        core_mapping_state = np.zeros(self.num_cores)
        if self.current_core_idx < (len(self.C)):
            core_mapping_state[self.current_core_idx] = 1.0
        # 任务核---通信节点映射矩阵,M[i][j]=1表示第i个任务核被映射到了第j个通信节点
        M = self.generate_M().flatten() 

        phase2_state = []
        if self.phase == 2:
            if self.current_comm_pair_idx >= len(self.com_pairs):
                phase2_state = np.zeros(self.phase2_state_dim, dtype=np.float32)
            else:
                src, dst = self.com_pairs[self.current_comm_pair_idx]
                path_link_vector = self.get_path_link_vector(src, dst)
                wavelength_state = np.zeros(len(self.communication_wavelengths))
                for i, wl in enumerate(self.communication_wavelengths):
                    if wl in self.assigned_wavelengths:
                        wavelength_state[i] = 1.0

                link_wl_matrix = self.get_link_wavelength_matrix().flatten()
                phase2_state = np.concatenate([
                                                path_link_vector,          # 当前通信用到哪些 link
                                                wavelength_state,          # 哪些波长已被占用
                                                link_wl_matrix             # 每条 link 上的波长占用
                                                ])
        else:
            phase2_state = np.zeros(self.phase2_state_dim, dtype=np.float32)
        state = np.concatenate([phase_flag, M, phase2_state])
        assert state.shape[0] == self.state_dim,f"State dim mismatch: got {state.shape[0]}, expected {self.state_dim}"

        return state
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        self.current_mapping = {}  # 当前映射方案 (core_name -> node_index)
        self.mapped_nodes = set()  # 已映射节点索引集合
        # 确保 available_nodes 是节点索引列表（0..num_nodes-1）
        self.available_nodes = list(range(self.num_nodes))
        self.current_core_idx = 0  # 当前要映射的核心索引
        self.phase = 1  # 当前阶段: 1=映射, 2=波长分配

        # 波长分配相关
        self.wavelength_assignment = {}  # pair -> wavelength (pair is (src_node, dst_node))
        self.assigned_wavelengths = set()
        self.current_comm_pair_idx = 0
        self.available_wavelengths = list(self.λ0 + i * (self.FSR / self.num_wavelengths) for i in range(self.num_wavelengths))

        # 记录一轮的奖励信息及其osnr
        self.record = {}
        self.reward_mapping = []
        self.reward_wavelength = []

        state = self._get_state()
        info = {}

        return state, info
    
    def step(self, action, Nettopology):
        """执行动作"""
        done = False
        reward = 0.0
        info = {}

        if self.phase == 1:  # 映射阶段
            
            # 动作是选择的节点索引
            node_selected = int(action)

            if node_selected >= self.num_nodes or node_selected not in self.available_nodes:
                # 无效动作惩罚
                reward = -2.0
                info['invalid_action'] = True
            else:
                core = self.C[self.current_core_idx]
                # 计算即时奖励（基于通信代价）
                reward_phase1 = self._calculate_mapping_reward(core, node_selected)
                reward += reward_phase1
                self.reward_mapping.append(reward_phase1)
                # 执行映射
                self.current_mapping[core] = node_selected

                self.mapped_nodes.add(node_selected)
                self.available_nodes.remove(node_selected)

                self.current_core_idx += 1

                # 检查是否完成映射
                if self.current_core_idx >= len(self.C):
                    self.phase = 2  # 进入波长分配阶段
                    self.com_pairs, self.com_pairs_links = self.get_paths_infomation()
                    # reward += 10.0  # 完成映射的奖励
                    info['phase_change'] = True

        elif self.phase == 2:  # 波长分配阶段
            # 动作是选择的波长索引
            wavelength_idx = int(action)
            # 首先判断有效性
            if wavelength_idx >= self.num_wavelengths or wavelength_idx < 0:
                reward = -2.0
                info['invalid_wavelength'] = True
            else:
                wavelength = self.λ0 + wavelength_idx * (self.FSR / self.num_wavelengths)

                if self.current_comm_pair_idx < len(self.com_pairs):
                    current_pair = self.com_pairs[self.current_comm_pair_idx]

                    if wavelength in self.assigned_wavelengths:
                        reward = -3.0
                        info['wavelength_conflict'] = True
                    else:

                        # 计算即时奖励
                        reward_phase2 = self._calculate_wavelength_reward(current_pair, wavelength)
                        reward += reward_phase2
                        self.reward_wavelength.append(reward_phase2)
                        # 分配波长
                        self.wavelength_assignment[current_pair] = wavelength

                        self.assigned_wavelengths.add(wavelength)
                        self.available_wavelengths.remove(wavelength)

                        self.current_comm_pair_idx += 1

                        # 检查是否完成所有分配
                        if self.current_comm_pair_idx >= len(self.com_pairs):
                            # 计算最终OSNR奖励
                            ONIs = self.get_ONI_models()
                            final_osnr,OSNRs,OSNRs_each_pairs = self._calculate_final_osnr(ONIs,self.wavelength_assignment,Nettopology)
                            reward += final_osnr   # OSNR作为主要奖励
                            done = True
                            self.record['reward_mapping'] = self.reward_mapping
                            self.record['reward_wavelength'] = self.reward_wavelength
                            self.record['osnr'] = final_osnr
                            self.record['OSNRs'] = OSNRs
                            self.record['OSNRs_each_pairs'] = OSNRs_each_pairs

        # 获取新状态
        next_state = self._get_state()

        # 添加时间惩罚（鼓励更快完成）
        # if not done:
        #     reward -= 0.1

        # Gymnasium 使用 terminated 和 truncated
        terminated = done
        truncated = False  # 可以根据需要设置截断条件

        return next_state, float(reward), bool(terminated), bool(truncated), info
    
    def _calculate_mapping_reward(self, core, node):#core->当前核心，node->选择的节点
        """计算映射阶段中将任务核core映射到节点node的奖励"""
        core_idx = self.C.index(core)
        reward = 0.5

        # 如果 current_mapping 之前为空（即第一个映射），给额外奖励
        if len(self.current_mapping) == 0:
            return reward 

        numerator = 0
        denominator = 0
        for mapped_core,mapped_node in self.current_mapping.items():
            mapped_idx = self.C.index(mapped_core)
            if self.W[core_idx][mapped_idx] == 0 and self.W[mapped_idx][core_idx] == 0:
                d_jk = 1
            elif self.W[core_idx][mapped_idx]:
                d_jk = 1 + self.distance_matrix[node][mapped_node]
            elif self.W[mapped_idx][core_idx]:
                d_jk = 1 + self.distance_matrix[mapped_node][node]
            numerator += d_jk
        for available_node in self.available_nodes:
            for mapped_core,mapped_node in self.current_mapping.items():
                mapped_idx = self.C.index(mapped_core)
                if available_node == node:
                    if self.W[core_idx][mapped_idx] == 0 and self.W[mapped_idx][core_idx] == 0:
                        d_jk = 1
                    elif self.W[core_idx][mapped_idx]:
                        d_jk = 1 + self.distance_matrix[node][mapped_node]
                    elif self.W[mapped_idx][core_idx]:
                        d_jk = 1 + self.distance_matrix[mapped_node][node]
                    denominator += d_jk
                else:
                    d_jk = 1
                    denominator += d_jk
        reward = numerator / denominator
        return float(reward)
    
    def _calculate_wavelength_reward(self, current_communication_pair, current_wavelength):
        _, paths_info = self.get_paths_infomation()
        current_link = paths_info[current_communication_pair]
        src, dst = current_communication_pair
        wl_reward = 1.0
        H = self.get_overlapped_paths(current_communication_pair, current_link, self.wavelength_assignment, paths_info)
        if not H:
            wl_reward += 1.0
        else:
            eps = 1e-6
            min_distance_current_wl = min(abs(current_wavelength - lam) for lam in H)
            min_distance_other_wl_sum = 0.0
            for wl in self.available_wavelengths:
                min_distance_other_wl = min(abs(wl - lam) for lam in H)
                min_distance_other_wl_sum += min_distance_other_wl
            denom = max(min_distance_other_wl_sum, eps)
            wl_reward += (min_distance_current_wl / denom)
        return float(wl_reward)

    def get_ONI_models(self):
        ONI_models = {}
        for i in range(len(self.ONI)):
            ONI_models[i] = []
            for j in range(len(self.communication_wavelengths)):
                mrj = MR(self.communication_wavelengths[j])
                ONI_models[i].append(mrj)
        return ONI_models
    
    def _calculate_final_osnr(self,ONI_models,wavelength_assignment,Nettopology,Ptx=6.11,lp=0.93897,lb=0.99885):
        """计算最终的OSNR。"""
        OSNRs = []
        OSNRs_each_pairs = {}
        _, paths_info = self.get_paths_infomation()
        ONI_models_copy = copy.deepcopy(ONI_models)
        #为ONI配置微环状态
        for src_dst,wl in wavelength_assignment.items():
            ONI_models_copy[src_dst[1]][self.communication_wavelengths.index(wl)].state = 1
        #为所有通信对计算OSNR
        for (src,dst), link in paths_info.items():
            current_wl = wavelength_assignment[(src,dst)]
            route_distance = dst - src if src < dst else len(self.ONI) - src + dst
            num_bends = self.get_num_bends(link,Nettopology)
            fai_signal_total = self.get_fai_signal_total(ONI_models_copy,current_wl,link)
            OP_signal = Ptx * fai_signal_total * (lp ** route_distance) * (lb ** num_bends)
            OP_crosstalk = 1e-12
            for (other_src,other_dst), other_link in paths_info.items():
                P_crosstalk  = 0
                if (other_src,other_dst) == (src,dst):
                    continue
                other_current_wl = wavelength_assignment[(other_src,other_dst)]
                if dst in [link[1] for link in other_link]:
                    if other_dst == dst:
                        if self.communication_wavelengths.index(other_current_wl) < self.communication_wavelengths.index(current_wl):
                            P_crosstalk = 0
                            OP_crosstalk += P_crosstalk
                        else:
                            route_distance = dst - other_src if dst > other_src else len(self.ONI) - (other_src - dst)
                            new_path = list(range(other_src, dst + 1)) if other_src < dst else list(range(other_src, len(self.ONI))) + list(range(0, dst +1))
                            new_link_info = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                            num_bends = self.get_num_bends(new_link_info,Nettopology)
                            fai_total_crosstalk = self.get_fai_crosstalk_total(ONI_models_copy,current_wl,other_current_wl,new_link_info)
                            P_crosstalk = Ptx * (lp ** route_distance) * (lb ** num_bends) * fai_total_crosstalk
                            OP_crosstalk += P_crosstalk
                    else:
                        route_distance = dst - other_src if dst > other_src else len(self.ONI) - (other_src - dst)
                        new_path = list(range(other_src, dst + 1)) if other_src < dst else list(range(other_src, len(self.ONI))) + list(range(0, dst +1))
                        new_link_info = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                        num_bends = self.get_num_bends(new_link_info,Nettopology)
                        fai_total_crosstalk = self.get_fai_crosstalk_total(ONI_models_copy,current_wl,other_current_wl,new_link_info)
                        P_crosstalk = Ptx * (lp ** route_distance) * (lb ** num_bends) * fai_total_crosstalk
                        OP_crosstalk += P_crosstalk
                else:
                    P_crosstalk = 0
                    OP_crosstalk += P_crosstalk
            OSNR = 10 * np.log10(OP_signal / OP_crosstalk)
            if OSNR > 65:
                OSNR = 65
            OSNRs.append(OSNR)
            OSNRs_each_pairs['({}->{})'.format(src,dst)] = OSNR
        avg_OSNR = sum(OSNRs) / len(OSNRs)
        return avg_OSNR,OSNRs,OSNRs_each_pairs
    
    def get_fai_signal_total(self,ONI_models_copy,current_wl, link_info):
        #current_wl为当前通信对的通信波长
        #link_info为当前通信对的路径信息
        fai_total = 1                 #初始化透射率
        for link in link_info:          #links = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
            dst_ONI = link[1]           #获取目的节点的ONI
            if link != link_info[-1]:           
                for mr in ONI_models_copy[dst_ONI]:#计算路径中经过的ONI的MR的through端透射率
                    xita = mr.theta(current_wl)
                    # print(f'the λ_res of mr{ONI_models_copy[dst_ONI].index(mr)} of ONI{dst_ONI} is {mr.λ_res}, state is {mr.state},fai_t={mr.fai_t(current_wl,xita)}')
                    fai_t = mr.fai_t(current_wl,xita)
                    fai_total *= fai_t
                # print(f'the fai_t_total of ONI{dst_ONI} of link {link} is {fai_t_total}')
            else:
                for mr in ONI_models_copy[dst_ONI]:
                    if mr.λ_res != current_wl:#计算最后一个ONI中目标微环之前微环的through端透射率
                        xita = mr.theta(current_wl)
                        # print(f'the λ_res of mr{ONI_models_copy[dst_ONI].index(mr)} of ONI{dst_ONI} is {mr.λ_res}, state is {mr.state},fai_t={mr.fai_t(current_wl,xita)}')
                        fai_t = mr.fai_t(current_wl,xita)
                        fai_total *= fai_t
                    elif mr.λ_res == current_wl:#计算最后一个ONI中目标微环的drop端透射率
                        xita = mr.theta(current_wl)
                        fai_d = mr.fai_d(current_wl,xita) 
                        fai_total *= fai_d ** 2     #此处计算了信号发射对应的微环的drop率
                        break
                # print(f'the fai_t_total of last ONI{dst_ONI} of link {link} is {fai_t_total}')
        # print(f'the final fai_t_total of src={src},dst={dst} is {fai_t_total}')
        return fai_total
    
    def get_fai_crosstalk_total(self,ONI_models_copy,current_wl,other_current_wl, link_info):
        #current_wl为要计算的串扰通信对对应的目标通信对的波长
        #other_current_wl为串扰通信对的通信波长
        #link_info为当前串扰通信对的路径信息
        fai_total = 1                 #初始化透射率
        for link in link_info:          #links = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
            dst_ONI = link[1]           #获取目的节点的ONI
            if link != link_info[-1]:           
                for mr in ONI_models_copy[dst_ONI]:#计算路径中经过的ONI的MR的through端透射率
                    xita = mr.theta(other_current_wl)
                    fai_t = mr.fai_t(other_current_wl,xita)
                    fai_total *= fai_t
            else:
                for mr in ONI_models_copy[dst_ONI]:
                    if mr.λ_mr != current_wl:#判断当前MR是否为目标信号的MR，若不是计算串扰信号经过该MR的through端透射率
                        xita = mr.theta(other_current_wl)
                        fai_t = mr.fai_t(other_current_wl,xita)
                        fai_total *= fai_t
                    elif mr.λ_mr == current_wl:#计算最后一个ONI中目标微环的drop端透射率
                        xita = mr.theta(other_current_wl)
                        fai_d = mr.fai_d(other_current_wl,xita)
                        fai_total *= fai_d
                        break
                for mr in ONI_models_copy[dst_ONI]:
                    if mr.λ_mr == other_current_wl:
                        get_state = mr.state
                        if get_state == 1:
                            xita = mr.theta(other_current_wl)
                            fai_tx_drop = mr.fai_d(other_current_wl,xita)#计算激光发射功率对应微环的drop端透射率
                            fai_total *= fai_tx_drop
                            break
                        else:
                            mr.state = 1
                            xita = mr.theta(other_current_wl)
                            fai_tx_drop = mr.fai_d(other_current_wl,xita)#计算激光发射功率对应微环的drop端透射率
                            fai_total *= fai_tx_drop
                            mr.state = 0
                            break
        return fai_total
    
    def render(self, mode="human"):
        """显示当前状态"""
        print(f"Phase: {self.phase}")
        print(f"Current Mapping: {self.current_mapping}")
        print(f"Mapped Nodes: {self.mapped_nodes}")
        if self.phase == 2:
            print(f"Wavelength Assignment: {self.wavelength_assignment}")

    def generate_M(self):
        M = np.zeros((len(self.C), len(self.ONI)))
        for core in self.C:
            if core in self.current_mapping:
                node = self.current_mapping[core]
                M[self.C.index(core)][node] = 1  # 生成映射矩阵M
        return M
    
    def generate_A(self,M):
        A = np.dot(M.T, np.dot(self.W, M))
        return A
    
    def get_paths_infomation(self):
        M = self.generate_M()
        A = self.generate_A(M)
        communication_pairs_info = []     #communication_pairs_info = [(3,7,4),(6,2,4),(2,3,1),...]
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if A[i][j]!= 0:
                    if i < j:
                        l = j-i
                        communication_pairs_info.append((i,j,l))
                    else:
                        l = len(self.ONI)-i+j
                        communication_pairs_info.append((i,j,l))
        #按路径长度进行降序排列
        communication_pairs_info.sort(key=lambda x:x[2],reverse=True)
        #更新communication_pairs_info,删除列表中每个元素的第三个元素
        for i in range(len(communication_pairs_info)):#update communication_pairs_info [(6,3),(3,7)...]
            communication_pairs_info[i] = (communication_pairs_info[i][0],communication_pairs_info[i][1])
        paths_info = {}      #paths_info = {(6,3):[(6,7),(7,0),(0,1),(1,2),(2,3)],(3,7):[(3,4),(4,5),(5,6),(6,7)]...}
        for (src,dst) in communication_pairs_info:
            path = list(range(src,dst+1)) if src < dst else list(range(src,len(self.ONI))) + list(range(0,dst+1))
            links = [(path[i],path[i+1]) for i in range(len(path)-1)]
            paths_info[(src,dst)] = links
        # return communication_pairs_info,paths_info
        return communication_pairs_info, paths_info
    
    def get_overlapped_paths(self,current_communication_pairs,current_link,p1_solution,paths_info):
        #paths_info,用于获取链路信息
        #p1_solution,用于检查重叠路径
        H = []     #接收重叠路径的波长
        #若p1_solution为空,则首次为路径分配波长,无重叠路径
        if not p1_solution:
            return H
        current_destination = current_communication_pairs[1]
        for (us,ud), λ in p1_solution.items():
            ulink = paths_info[(us,ud)]
            ulink_destination = ulink[-1][1]
            dst_in_ulink = any(current_destination == node[1] for node in ulink)
            ulink_dst_in_current_link = any(ulink_destination == node[1] for node in current_link)
            if dst_in_ulink or ulink_dst_in_current_link:
                H.append(λ)
        return H
    
    def get_distance(self,wl,H_current):
        #此函数用于获取最小波长间隔
        #wl为available_wavelengths中的可用波长
        if not H_current :
            return 1
        #distance等于wl与H_current中波长差的绝对值的最小值
        min_distance = float('inf')
        for λ in H_current:
            diff = abs(wl - λ)
            min_distance = min(min_distance,diff)
        return min_distance

    def get_num_bends(
        self,
        link_info: Dict[Tuple[int, int], List[Tuple[int, int]]],
        calc_type: str  # 移除默认值，必须显式传入
    ) -> int:
        # 严格校验参数合法性
        valid_calc_types = ["pip", "mwd", "mp3dec", "mp3enc"]
        if calc_type not in valid_calc_types:
            raise ValueError(
                f"calc_type必须是{valid_calc_types}中的一种，当前传入：{calc_type}"
            )
        
        # pip对应的计算逻辑
        if calc_type == "pip":
            has_34 = (3, 4) in link_info
            has_70 = (7, 0) in link_info
            if has_34 and has_70:
                return 4
            elif has_34 or has_70:
                return 2
            else:
                return 0
        
        # mwd对应的计算逻辑
        elif calc_type == "mwd" or calc_type == "mp3dec":
            target_segments = [(2, 3), (5, 6), (8, 9), (11, 0)]
            count = 0
            for segment in link_info:
                if segment in target_segments:
                    count += 1
            return count
        
        elif  calc_type == "mp3enc":
            target_segments = [(2, 3), (5, 6), (8, 9), (12, 0)]
            count = 0
            for segment in link_info:
                if segment in target_segments:
                    count += 1
            return count
    
    def get_pairs_wl_matrix(self):
        #生成通信对与波长分配矩阵
        pairs_wl_matrix = np.zeros((len(self.com_pairs),len(self.communication_wavelengths)))
        for i,(src,dst) in enumerate(self.com_pairs):
            if (src,dst) in self.wavelength_assignment:
                wl = self.wavelength_assignment[(src,dst)]
                wl_index = self.communication_wavelengths.index(wl)
                pairs_wl_matrix[i][wl_index] = wl
        return pairs_wl_matrix
    
    def get_path_link_vector(self, src, dst):
        """
        返回当前通信对的 link-level path vector
        """
        vec = np.zeros(self.num_links, dtype=np.float32)
        for link in self.com_pairs_links[(src, dst)]:
            if link in self.link_id_map:
                vec[self.link_id_map[link]] = 1.0
        return vec

    def get_link_wavelength_matrix(self):
        """
        link_wl[i][j] = 1
        表示第 i 条 link 上已经使用了第 j 个波长
        """
        mat = np.zeros((self.num_links, len(self.communication_wavelengths)), dtype=np.float32)
        for (src, dst), wl in self.wavelength_assignment.items():
            wl_idx = self.communication_wavelengths.index(wl)
            for link in self.com_pairs_links[(src, dst)]:
                link_id = self.link_id_map[link]
                mat[link_id][wl_idx] = 1.0
        return mat
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_features = self.shared_network(x)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value
    
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 clip_epsilon=0.2, update_epochs=4, batch_size=64, lam=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma#折扣因子
        self.lam = lam#GAE参数，权衡方差与偏差
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # memory: list of tuples (state, action, log_prob, value, reward, done)
        # where log_prob and value are scalars (floats)
        self.memory = []

    def select_action(self, state, valid_action_mask: List[bool] = None):
        """
        选择动作。可选 valid_action_mask：长度等于 action_dim 的布尔列表，表示哪些动作当前有效。
        如果提供 mask，会把无效动作的 logits 置为很小的值，达到 action masking 效果。
        返回: action(int), log_prob(float), value(float)
        """
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        logits, value = self.policy(state_tensor)  # logits: [1, action_dim] value:当前状态价值估值
        '''能否实现在进行神经网络前向传播时，直接将无效动作的输出置为一个极小值，从而避免选择这些动作？'''
        if valid_action_mask is not None:
            mask = torch.BoolTensor(valid_action_mask).to(self.device)
            # 若全部为 False（不该发生），允许全部 True 作为后备
            if mask.sum().item() == 0:
                mask = torch.ones_like(mask)
            minus_inf = -1e9
            logits = logits.masked_fill(~mask.unsqueeze(0), minus_inf)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action).cpu().item()
        value_item = value.squeeze().cpu().item()

        return int(action.item()), float(log_prob), float(value_item)

    def store_transition(self, state, action, log_prob, value, reward, done, phase, mask):
        self.memory.append((state,int(action),float(log_prob),float(value),float(reward),float(done),int(phase),np.array(mask, dtype=np.bool_)))

    def _compute_gae(self, rewards, values, dones, phases):
        #直接用总回报减价值（MC 方法）：方差大（受随机奖励影响），但无偏；
        #用单步 TD 误差（TD (0)）：方差小，但有偏；
        #在方差和偏差之间做最优权衡，得到更稳定、更准确的优势值，让 PPO 的更新更高效。”
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        advantages = torch.zeros(T, device=self.device)

        last_gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            is_terminal = dones[t]

            # ⭐ Phase-aware cut ⭐
            if t < T - 1 and phases[t] != phases[t + 1]:
                next_value = 0.0
                last_gae = 0.0

            mask = 1.0 - is_terminal
            delta = rewards[t] + self.gamma * next_value * mask - values[t]#计算单步时序误差
            last_gae = delta + self.gamma * self.lam * mask * last_gae

            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        next_value = values[t]

        return returns, advantages

    def update(self):
        if len(self.memory) == 0:
            return

        # 转换数据为 tensor
        states, actions, old_log_probs, values, rewards, dones, phases, masks= zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        phases = torch.LongTensor(phases).to(self.device)
        masks = torch.BoolTensor(np.array(masks)).to(self.device)

        # compute returns and advantages via GAE
        returns, advantages = self._compute_gae(rewards, values, dones, phases)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(states)
        for epoch in range(self.update_epochs):#更新周期，一批数据更新四次
            indices = torch.randperm(N).to(self.device)
            for start in range(0, N, self.batch_size):#从第0个数据到第N个数据，每次步长为batch_size
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                logits, new_values = self.policy(batch_states)
                batch_masks = masks[batch_idx]
                minus_inf = -1e9
                logits = logits.masked_fill(~batch_masks, minus_inf)
                new_probs = torch.softmax(logits, dim=-1)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (new_values.squeeze() - batch_returns).pow(2).mean()
                entropy = dist.entropy().mean()
                entropy_bonus = 0.01 * entropy

                loss = actor_loss + 0.5 * critic_loss - entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        # 清空 memory
        self.memory = []
    
def construct_action_mask(env: RLOpticalNetworkEnv):
    """
    为当前 env 状态构造 action mask (True 表示该动作有效)
    - Phase1 (映射): 动作代表 node index; 有效 iff node in available_nodes
    - Phase2 (波长分配): 动作代表 wavelength idx; 有效 iff idx < num_wavelengths 且对应波长未被使用
    mask 长度为 env.action_space.n
    """
    n_actions = env.action_space.n
    mask = [False] * n_actions

    if env.phase == 1:
        for i in range(n_actions):
            if i < env.num_nodes and (i in env.available_nodes):
                mask[i] = True
    else:
        for i in range(n_actions):
            if i < env.num_wavelengths:
                wl = env.λ0 + i * (env.FSR / env.num_wavelengths)
                if wl not in env.assigned_wavelengths:
                    mask[i] = True
    # 如果全部为 False，允许所有动作以免死锁（通常不应发生）
    if not any(mask):
        mask = [True] * n_actions
    return mask
