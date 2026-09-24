from typing import Dict, List, Tuple
import numpy as np
from . import MR
import copy
import joblib
import os

class AntColonySystem:
    def __init__(self,
                com_wavelengths: List[float],
                ONI: List[int],
                num_ants: int = 15,
                iterations: int =2000 ,
                alpha: float = 2,beta: float = 2.0,
                rho: float = 0.5,Q: float = 60,K: int = 150,
                tao_min: float = 0.5,
                tao_max: float = 2,
                ) -> None:
        self.com_wavelengths = com_wavelengths
        self.ONI = ONI
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.K = K
        self.tao_min = tao_min
        self.tao_max = tao_max
    
    #前n项概率求和
    def get_sum_n(self,prob: List[float], n: int) -> float:
        if n > len(prob):
            raise ValueError("n 超出列表长度")
        return sum(prob[:n])
    
    #应用映射
    def apply_mapping(self, C: List[str], 
                    W: np.ndarray,
                    dis_matrix: np.ndarray,
                    pheromone: np.ndarray) -> Dict[str, int]:
        """
        生成一个 ant mapping（原始实现，不包含 ML 预测/重试）
        """
        ant_soluton = {}
        mapped_nodes = set()
        for core in C:
            available_nodes = [node for node in self.ONI if node not in mapped_nodes]
            etas = []
            if not ant_soluton:  # first mapping: heuristic = 1
                for node in available_nodes:
                    etas.append(1)
            else:
                for node in available_nodes:
                    numerator = 0
                    denominator = 0
                    for mapped_core in ant_soluton.keys():
                        mapped_node = ant_soluton[mapped_core]
                        if W[C.index(core)][C.index(mapped_core)] == 0 and W[C.index(mapped_core)][C.index(core)] == 0:
                            d_jk = 1
                        elif W[C.index(core)][C.index(mapped_core)]:
                            d_jk = 1 + dis_matrix[node][mapped_node]
                        elif W[C.index(mapped_core)][C.index(core)]:
                            d_jk = 1 + dis_matrix[mapped_node][node]
                        numerator += d_jk
                    for available_node in available_nodes:
                        for mapped_core in ant_soluton.keys():
                            mapped_node = ant_soluton[mapped_core]
                            if available_node == node:
                                if W[C.index(core)][C.index(mapped_core)] == 0 and W[C.index(mapped_core)][C.index(core)] == 0:
                                    d_jk = 1
                                elif W[C.index(core)][C.index(mapped_core)]:
                                    d_jk = 1 + dis_matrix[node][mapped_node]
                                elif W[C.index(mapped_core)][C.index(core)]:
                                    d_jk = 1 + dis_matrix[mapped_node][node]
                                denominator += d_jk
                            else:
                                denominator += 1
                    if denominator == 0:
                        eta = 0
                    else:
                        eta = numerator / denominator
                    etas.append(eta)
            info = []
            for i, node in enumerate(available_nodes):
                pheromone_value = pheromone[C.index(core)][node] ** self.alpha
                heuristic_value = (etas[i] ** self.beta) if etas[i] != 0 else 0
                info.append(pheromone_value * heuristic_value)
            info = np.array(info)
            if info.sum() == 0:
                probabilities = np.ones_like(info) / len(info)
            else:
                probabilities = info / info.sum()
            Prob = [0] * len(available_nodes)
            for j in range(len(available_nodes)):
                Prob[j] = self.get_sum_n(probabilities, j + 1)
            random_num = np.random.uniform()
            selected_node = available_nodes[-1]
            for i in range(len(Prob)):
                if random_num < Prob[i]:
                    selected_node = available_nodes[i]
                    break
            ant_soluton[core] = selected_node
            mapped_nodes.add(selected_node)

        return ant_soluton
    
    #应用映射with machine learning prediction
    def apply_mapping_ml_predict(self, C: List[str], 
                      W: np.ndarray,
                      dis_matrix: np.ndarray,
                      pheromone: np.ndarray) -> Dict[str, int]:
        """
        Generate an ant mapping. If an XGBoost model exists it will predict the mapping's
        OSNR; mappings with predicted OSNR < 40 dB are retried (up to max_attempts).
        """
        # Try loading a saved XGBoost model (joblib). prefer tuned model then fallback.
        model = None
        feature_length = None
        candidate_paths = [
            os.path.join('RL_ACO_compare', 'ONoC_Optimization', 'ml', 'model_xgb_mapping.joblib')
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    loaded = joblib.load(p)
                    if isinstance(loaded, dict):
                        model = loaded.get('model', None)
                        feature_length = loaded.get('feature_length', None)
                    else:
                        model = loaded
                    break
                except Exception:
                    model = None

        max_attempts = 10
        attempt = 0

        def build_feature_from_solution(sol: Dict[str, int], cores: List[str]):
            # Use sorted core keys to match training parsing
            cores_sorted = sorted(cores)
            vals = [sol.get(c, 0) for c in cores_sorted]
            arr = np.asarray(vals, dtype=float)
            if feature_length is not None:
                if arr.size > feature_length:
                    arr = arr[:feature_length]
                elif arr.size < feature_length:
                    arr = np.pad(arr, (0, feature_length - arr.size), 'constant')
            return arr.reshape(1, -1)

        while True:
            attempt += 1
            ant_soluton = {}
            mapped_nodes = set()
            for core in C:
                available_nodes = [node for node in self.ONI if node not in mapped_nodes]
                etas = []
                if not ant_soluton:  # first mapping: heuristic = 1
                    for node in available_nodes:
                        etas.append(1)
                else:
                    for node in available_nodes:
                        numerator = 0
                        denominator = 0
                        for mapped_core in ant_soluton.keys():
                            mapped_node = ant_soluton[mapped_core]
                            if W[C.index(core)][C.index(mapped_core)] == 0 and W[C.index(mapped_core)][C.index(core)] == 0:
                                d_jk = 1
                            elif W[C.index(core)][C.index(mapped_core)]:
                                d_jk = 1 + dis_matrix[node][mapped_node]
                            elif W[C.index(mapped_core)][C.index(core)]:
                                d_jk = 1 + dis_matrix[mapped_node][node]
                            numerator += d_jk
                        for available_node in available_nodes:
                            for mapped_core in ant_soluton.keys():
                                mapped_node = ant_soluton[mapped_core]
                                if available_node == node:
                                    if W[C.index(core)][C.index(mapped_core)] == 0 and W[C.index(mapped_core)][C.index(core)] == 0:
                                        d_jk = 1
                                    elif W[C.index(core)][C.index(mapped_core)]:
                                        d_jk = 1 + dis_matrix[node][mapped_node]
                                    elif W[C.index(mapped_core)][C.index(core)]:
                                        d_jk = 1 + dis_matrix[mapped_node][node]
                                    denominator += d_jk
                                else:
                                    denominator += 1
                        if denominator == 0:
                            eta = 0
                        else:
                            eta = numerator / denominator
                        etas.append(eta)
                info = []
                for i, node in enumerate(available_nodes):
                    pheromone_value = pheromone[C.index(core)][node] ** self.alpha
                    heuristic_value = (etas[i] ** self.beta) if etas[i] != 0 else 0
                    info.append(pheromone_value * heuristic_value)
                info = np.array(info)
                if info.sum() == 0:
                    probabilities = np.ones_like(info) / len(info)
                else:
                    probabilities = info / info.sum()
                Prob = [0] * len(available_nodes)
                for j in range(len(available_nodes)):
                    Prob[j] = self.get_sum_n(probabilities, j + 1)
                random_num = np.random.uniform()
                selected_node = available_nodes[-1]
                for i in range(len(Prob)):
                    if random_num < Prob[i]:
                        selected_node = available_nodes[i]
                        break
                ant_soluton[core] = selected_node
                mapped_nodes.add(selected_node)

            # If no model, return the generated solution
            if model is None:
                return ant_soluton

            # build feature and predict
            try:
                feat = build_feature_from_solution(ant_soluton, C)
                pred_osnr = float(model.predict(feat)[0])
            except Exception:
                # on any failure, return current solution
                return ant_soluton

            if pred_osnr >= 40 or attempt >= max_attempts:
                return ant_soluton
            # else retry (loop continues)
    
    #应用映射矩阵
    def get_m(self,ant_solution: Dict[str, int], C: List[str]) -> np.ndarray:
        m = np.zeros((len(C), len(C)))
        for core in C:
            if core in ant_solution:
                node = ant_solution[core]
                m[C.index(core)][node] = 1
        return m
    
    #结构通信矩阵A
    def get_a(self,M: np.ndarray, W: np.ndarray) ->np.ndarray:
        return np.dot(M.T,np.dot(W,M))
    
    #波长分配deatination
    def wavelength_assignment_dest(self,A: np.ndarray) -> np.ndarray:
        available_wavelengths = self.com_wavelengths.copy()
        path_wavelengths = {}  #用于记录通信路径的波长分配情况
        #定义L矩阵，用于记录通信路径的波长分配情况
        L = np.zeros((len(self.ONI), len(self.ONI)))
        #遍历通信结构矩阵A，为每个通信对分配通信波长
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if A[i][j] > 0:
                    # 检查路径的终点节点j是否已经存在于path_wavelengths字典中
                    if j not in path_wavelengths:
                        # 如果没有，则从available_wavelengths列表中取出第一个可用波长
                        path_wavelengths[j] = available_wavelengths.pop(0)
                    # 分配波长
                    L[i][j] = path_wavelengths[j]  # 更新L矩阵中的波长信息 
        #生成三个可返回参数：可用波长available_wavelengths、波长分配情况path_wavelengths、路径矩阵L
        return L
    
    #波长分配source
    def wavelength_assignment_source(self,A: np.ndarray) -> np.ndarray:
        available_wavelengths = self.com_wavelengths.copy
        # source-based wavelength assignment, according to A(结构通信矩阵)
        path_wavelengths = {}  # 用于记录通信路径的波长分配情况
        # 定义L矩阵，用于记录通信路径的波长分配情况
        L = np.zeros((len(self.ONI), len(self.ONI)))
        # 遍历通信结构矩阵A，为每个通信对分配通信波长
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if A[i][j] > 0:
                    # 检查路径的源节点节点i是否已经存在于path_wavelengths字典中
                    if i not in path_wavelengths:
                        # 如果没有，则从available_wavelengths列表中取出第一个可用波长
                        path_wavelengths[i] = available_wavelengths.pop(0)
                    # 分配波长
                    L[i][j] = path_wavelengths[i]
        # 生成三个可返回参数：可用波长available_wavelengths、波长分配情况path_wavelengths、路径矩阵L
        return L
    
    #波长分配 order communication based
    def wavelength_assignment_communication_order(self,A: np.ndarray) -> np.ndarray:
        available_wavelengths = self.com_wavelengths.copy
        length_wa = len(available_wavelengths) - 2
        # 定义L矩阵，用于记录通信路径的波长分配情况
        L = np.zeros((len(self.ONI), len(self.ONI)))
        # 基于communication-based 分配波长
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if i == j:
                    L[i][j] = 0
                elif i < j:
                    L[i][j] = available_wavelengths[j - i]
                    L[j][i] = available_wavelengths[length_wa - (j - i)]
        # 遍历通信结构矩阵A，更新波长分配
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if A[i][j] == 0:
                    L[i][j] = 0
                else:
                    continue
        return L
        
    #波长分配 inverse communication based
    def wavelength_assignment_communication_inverse(self,A: np.ndarray) -> np.ndarray:
        available_wavelengths = self.com_wavelengths.copy
        length_wa = len(available_wavelengths)
        # 定义L矩阵，用于记录通信路径的波长分配情况
        L = np.zeros((len(self.ONI), len(self.ONI)))
        # 基于communication-based 分配波长
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if i == j:
                    L[i][j] = 0
                elif i < j:
                    L[i][j] = available_wavelengths[length_wa - (j - i)]
                    L[j][i] = available_wavelengths[j - i]
        # 遍历通信结构矩阵A，更新波长分配
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if A[i][j] == 0:
                    L[i][j] = 0
                else:
                    continue
        return L
    
    #为通信对生成链路信息
    def generate_link_info(self,L: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        paths = {}
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if L[i][j] > 0:
                    path = list(range(i,j+1)) if i < j else list(range(i,len(self.ONI))) + list(range(0,j+1))#path = [0,1,2,3,4,5,6,7]
                    links = [(path[i],path[i+1]) for i in range(len(path)-1)]   #links = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
                    paths [(i,j)] = links
        return paths
    
    #计算弯曲次数pip
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
    
    #OP_signal总透射率
    def get_fai_signal_total(self,
                            ONI_models_copy: Dict[int, List[MR]], 
                            current_wl: float, 
                            link_info: List[Tuple[int, int]]) -> float:
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
                        fai_total *= fai_d ** 2
                        break
        return fai_total
    
    #OP_crosstalk总透射率
    def get_fai_crosstalk_total(self,
                               ONI_models_copy: Dict[int, List[MR]], 
                               current_wl: float,
                               other_current_wl:float,
                               link_info: List[Tuple[int, int]]) -> float:
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
    
    #OSNR
    def get_osnr_mapping(self,
                        ONI_models: Dict[int, List[MR]], 
                        L: np.ndarray,
                        net_topology: str, 
                        P_tx: float = 6.11,
                        lp: float = 0.93897,
                        lb: float = 0.99885) -> Tuple[float, Dict[Tuple[int, int], float]]:
        OSNRs = {}  #存储当前ant_solution的每个通信对的OSNR
        #为通信对生成链路信息
        paths = self.generate_link_info(L)
        ONI_models_copy = copy.deepcopy(ONI_models)
        for src_dst in paths.keys():
            src = src_dst[0]
            dst = src_dst[1]
            current_λ = L[src][dst]#获取当前通信对的通信波长
            try:
                ONI_models_copy[dst][self.com_wavelengths.index(current_λ)].state = 1             #设置目的节点对应photodector微环的谐振state为1
            except ValueError:
                print(f"Warning: Wavelength {current_λ} not found in available wavelengths.")  
        #calculate the OSNR for each communication pair
        for (src_dst,link_info) in paths.items():
            src = src_dst[0]
            dst = src_dst[1] 
            current_wl = L[src][dst]  #获取当前通信对的通信波长   
            route_distance = dst - src if dst > src else len(self.ONI) - (src - dst)
            num_bends = self.get_num_bends(link_info,net_topology)  
            #计算当前通信路径经历的ONIs中MR的through端的透射率
            fai_total = self.get_fai_signal_total(ONI_models_copy,current_wl,link_info)
            OP_signal = P_tx * (lp ** route_distance) * (lb **num_bends) * fai_total
            #计算当前通信对的串扰噪声
            OP_crosstalk = 1e-12 #初始化总的串扰功率
            for (other_src_dst,other_link_info) in paths.items():
                P_crosstalk = 0  #初始化当前串扰链路的串扰功率
                if other_src_dst == src_dst:
                    continue
                if other_src_dst[1] == dst:
                    continue
                #获取other_src_dst的通信波长
                other_src = other_src_dst[0]
                other_dst = other_src_dst[1]
                other_current_wl = L[other_src][other_dst]
                #判断串扰链路是否经过dst节点
                if dst in [link[1] for link in other_link_info]:    #判断串扰信号链路是否经过dst节点
                    if other_dst == dst:        #判断串扰信号链路的dst是否是目标信号的dst
                        if self.com_wavelengths.index(other_current_wl) < self.com_wavelengths.index(current_wl):
                            P_crosstalk = 0
                            OP_crosstalk += P_crosstalk
                        else:
                            '''计算从other_src到dst的串扰信号功率'''
                            route_distance = dst - other_src if dst > other_src else len(self.ONI) - (other_src - dst)
                            #更新other_link_info为从other_src到dst的链路信息
                            new_path = list(range(other_src, dst + 1)) if other_src < dst else list(range(other_src, len(self.ONI))) + list(range(0, dst +1))
                            new_link_info = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                            num_bends = self.get_num_bends(new_link_info,net_topology)  # 计算弯曲次数
                            fai_total_crosstalk = self.get_fai_crosstalk_total(ONI_models_copy,current_wl,other_current_wl,new_link_info)
                            P_crosstalk = P_tx * (lp ** route_distance) * (lb ** num_bends) * fai_total_crosstalk
                            OP_crosstalk += P_crosstalk
                    else:   #目标信号的dst不是串扰信号链路的dst
                        #计算从other_src到dst的串扰信号功率
                        route_distance = dst - other_src if dst > other_src else len(self.ONI) - (other_src - dst)
                        #更新other_link_info为从other_src到dst的链路信息
                        new_path = list(range(other_src, dst + 1)) if other_src < dst else list(range(other_src, len(self.ONI))) + list(range(0, dst +1))
                        new_link_info = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                        # print(f'the new_link_info of other_src={other_src},dst={dst} is {new_link_info}')
                        num_bends = self.get_num_bends(new_link_info, net_topology)  # 计算弯曲次数
                        #计算new_link_info中每个ONI的MR的through端的透射率
                        fai_total_crosstalk = self.get_fai_crosstalk_total(ONI_models_copy,current_wl,other_current_wl,new_link_info)
                        P_crosstalk = P_tx * (lp ** route_distance) * (lb ** num_bends) * fai_total_crosstalk
                        OP_crosstalk += P_crosstalk
                else:    
                    P_crosstalk = 0
                    OP_crosstalk += P_crosstalk
            OSNR = 10 * np.log10(OP_signal / OP_crosstalk) 
            if OSNR > 65:
                OSNR = 65
            OSNRs[(src, dst)] = OSNR
        #计算平均值
        avg_OSNR = sum(OSNRs.values()) / len(OSNRs)
        return avg_OSNR, OSNRs
    
    #信息素更新
    def update_pheromone_mapping(self,
                                pheromone: np.ndarray,
                                C: List[str],
                                iteration: int,
                                best_osnrs_all_iterations: List[float],
                                current_iteration_solutions_of_all_ants: List[Dict[str, int]],
                                current_iteration_osnrs_of_all_ants: List[float],
                                elite_solution: Dict[str, int],
                                elite_osnr: float) -> np.ndarray:
        #elite_solution = global_best_solution
        #elite_osnr = global_best_osnr
        #动态计算挥发率
        def get_dynamic_rho(iteration: int , rho_max: float = 0.5, rho_min: float = 0.1) -> float:
            base_rho = rho_max * np.exp(-3.0 * iteration / self.iterations)
            if iteration > 10 and (best_osnrs_all_iterations[iteration] - best_osnrs_all_iterations[iteration-10]) < 0.1:
                base_rho = min(rho_max, base_rho * 1.2)
            return max(rho_min, base_rho)
        current_rho = get_dynamic_rho(iteration)
        #1.计算当前迭代所有蚂蚁OSNR的归一化权重
        elite_boost = 2.5 - 2.0 * (iteration / self.iterations)
        osnr_min, osnr_max = np.min(current_iteration_osnrs_of_all_ants), np.max(current_iteration_osnrs_of_all_ants)
        weights = [(osnr - osnr_min) / (osnr_max - osnr_min + 1e-6) for osnr in current_iteration_osnrs_of_all_ants]
        #2.所有蚂蚁加权更新
        for idx, ant_solution in enumerate(current_iteration_solutions_of_all_ants):
            weight = weights[idx]
            for i in range(len(C)):
                for j in range(len(self.ONI)):
                    #判断ant_solution中是否存在'C[i]' : j
                    if C[i] in ant_solution and ant_solution[C[i]] == j:
                        delta_tao = self.Q / (self.K - current_iteration_osnrs_of_all_ants[idx]) * weight
                    else:
                        delta_tao = 0
                    pheromone[i][j] = pheromone[i][j]*(1-current_rho) + delta_tao
        #3.精英解强化
        for i in range(len(C)):
            for j in range(len(self.ONI)):
                #判断ant_solutions中是否存在'C[i]' : j
                if C[i] in elite_solution and elite_solution[C[i]] == j:
                    delta_tao = self.Q / (self.K - elite_osnr) * elite_boost
                else:
                    delta_tao = 0
                pheromone[i][j] = pheromone[i][j]*(1-current_rho) + delta_tao
        pheromone = np.clip(pheromone, self.tao_min, self.tao_max)
        return pheromone

    #传统信息素更新
    def update_pheromone_mapping_tra(self,
                                    pheromone: np.ndarray,
                                    C: List[str],
                                    local_best_solution: Dict[str, int],
                                    local_best_osnr: float) -> np.ndarray:
        pheromone_old = pheromone
        for i in range(len(C)):
            for j in range(len(self.ONI)):
                #判断current_iteration_best_solution中是否存在'C[i]' : j
                if C[i] in local_best_solution and local_best_solution[C[i]] == j:
                    delta_tao = self.Q / (self.K - local_best_osnr) 
                else:
                    delta_tao = 0
                pheromone_old[i][j] = (1 - self.rho) * pheromone[i][j] + delta_tao
        pheromone_new = np.clip(pheromone_old, self.tao_min, self.tao_max)
        pheromone_delta = pheromone_new - pheromone
        # print(f"pheromone_delta = {pheromone_delta}")
        return pheromone_new
    #p2
    #解信息重现
    def solution_init(self,
                     best_solution_mapping: Dict[str, int],
                     C: List[str],
                     W: np.ndarray,) -> None:
        best_solution_M = self.get_m(best_solution_mapping,C)
        best_solution_A = self.get_a(best_solution_M,W)
        self.sorted_paths,self.sorted_paths_links = self.get_sorted_paths_assign(best_solution_A)

    #路径重排信息
    def get_sorted_paths_assign(self,best_solution_A: np.ndarray
                                ) -> Tuple[List[Tuple[int,int]],Dict[Tuple[int,int],List[Tuple[int,int]]]]:
        sorted_paths = []     #sorted_paths = [(3,7,4),(6,2,4),(2,3,1),...]
        for i in range(len(self.ONI)):
            for j in range(len(self.ONI)):
                if best_solution_A[i][j]!= 0:
                    if i < j:
                        l = j-i
                        sorted_paths.append((i,j,l))
                    else:
                        l = len(self.ONI)-i+j
                        sorted_paths.append((i,j,l))
        #按路径长度进行降序排列
        sorted_paths.sort(key=lambda x:x[2],reverse=True)
        #更新communication_pairs_info,删除列表中每个元素的第三个元素
        for i in range(len(sorted_paths)):
            sorted_paths[i] = (sorted_paths[i][0],sorted_paths[i][1])
        sorted_paths_links = {}      #sorted_paths_links = {(6,3):[(6,7),(7,0),(0,1),(1,2),(2,3)],(3,7):[(3,4),(4,5),(5,6),(6,7)]...}
        for (src,dst) in sorted_paths:
            path = list(range(src,dst+1)) if src < dst else list(range(src,len(self.ONI))) + list(range(0,dst+1))
            links = [(path[i],path[i+1]) for i in range(len(path)-1)]
            sorted_paths_links[(src,dst)] = links
        return sorted_paths,sorted_paths_links
    
    #获取当前路径的重叠路径波长
    def get_H(self,
              current_communication_pairs: Tuple[int,int],
              current_link: List[Tuple[int,int]],
              ant_solution_assign: Dict[Tuple[int,int], float],
              sorted_paths_links: Dict[Tuple[int,int],List[Tuple[int,int]]]) -> List[float]:
        #sorted_paths_links,用于获取链路信息
        #ant_solution_assign,用于检查重叠路径
        H = []     #接收重叠路径的波长
        #若ant_solution为空,则首次为路径分配波长,无重叠路径
        if not ant_solution_assign:
            return H
        current_destination = current_communication_pairs[1]
        for (us,ud), λ in ant_solution_assign.items():
            ulink = sorted_paths_links[(us,ud)]
            ulink_destination = ulink[-1][1]
            dst_in_ulink = any(current_destination == node[1] for node in ulink)
            ulink_dst_in_current_link = any(ulink_destination == node[1] for node in current_link)
            if dst_in_ulink or ulink_dst_in_current_link:
                H.append(λ)
        return H
    
    #获取当可用波长与重叠波长的最小波长间距
    def get_spacing(self,
                   wl: float,
                   H_current: List[float]):
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
    
    #波长分配
    def wavelength_assignment(self,pheromone: np.ndarray) -> Dict[Tuple[int,int],float]:
        ant_solution = {}
        used_wavelengths = set()
        for (src,dst), link in self.sorted_paths_links.items():
            available_wavelengths = [wavelength for wavelength in self.com_wavelengths if wavelength not in used_wavelengths]
            current_communication_pairs = (src,dst)
            current_link = link
            H_current = self.get_H(current_communication_pairs,current_link,ant_solution,self.sorted_paths_links)
            distances = []
            #在available_wavelengths中为每个wl计算其与H_current中记录的波长的最小间隔
            for wl in available_wavelengths:
                current_wl_distance = self.get_spacing(wl,H_current)
                distances.append(current_wl_distance)
            distances = np.array(distances)
            #归一化波长间隔为启发式信息
            etas = distances / distances.sum()
            info = []
            for i in range(len(available_wavelengths)):
                pheromone_value = pheromone[self.sorted_paths.index((src,dst))][self.com_wavelengths.index(available_wavelengths[i])] ** self.alpha
                heuristic_value = etas[i] ** self.beta
                info.append(pheromone_value * heuristic_value)
            info = np.array(info)
            #归一化获得为当前路径映射到每一可用波长的概率
            info = info / info.sum()
            Prob = [0] * len(available_wavelengths)
            for i in range(len(available_wavelengths)):
                Prob[i] = self.get_sum_n(info,i+1)
            random_num = np.random.uniform()
            for j in range(len(available_wavelengths)):
                if random_num < Prob[j]:
                    ant_solution[(src,dst)] = available_wavelengths[j]
                    used_wavelengths.add(available_wavelengths[j])
                    available_wavelengths.remove(available_wavelengths[j])
                    break
        return ant_solution
    
    #
    def get_osnr_assign(self,
                       ant_solution_assign: Dict[Tuple[int,int],float],
                       ONI_models: Dict[int, List[MR]],
                       net_topology: str,
                       Ptx: float = 6.11,
                       lp: float = 0.93897,
                       lb: float = 0.99885):
        OSNRs = {}
        ONI_models_copy = copy.deepcopy(ONI_models)
        #according to the ant_soluton,configure the ONI models corresponding to the communication pairs
        for src_dst,wl in ant_solution_assign.items():
            ONI_models_copy[src_dst[1]][self.com_wavelengths.index(wl)].state = 1
        #calculate osnr for each communication pair
        for (src,dst), link in self.sorted_paths_links.items():
            current_wl = ant_solution_assign[(src,dst)]
            route_distance = dst - src if src < dst else len(self.ONI) - src + dst
            num_bends = self.get_num_bends(link, net_topology)
            fai_signal_total = self.get_fai_signal_total(ONI_models_copy,current_wl,link)
            OP_signal = Ptx * fai_signal_total * (lp ** route_distance) * (lb ** num_bends)
            OP_crosstalk = 1e-12
            for (other_src,other_dst), other_link in self.sorted_paths_links.items():
                P_crosstalk  = 0
                if (other_src,other_dst) == (src,dst):
                    continue
                other_current_wl = ant_solution_assign[(other_src,other_dst)]
                if dst in [link[1] for link in other_link]:
                    if other_dst == dst:
                        if self.com_wavelengths.index(other_current_wl) < self.com_wavelengths.index(current_wl):
                            P_crosstalk = 0
                            OP_crosstalk += P_crosstalk
                        else:
                            route_distance = dst - other_src if dst > other_src else len(self.ONI) - (other_src - dst)
                            new_path = list(range(other_src, dst + 1)) if other_src < dst else list(range(other_src, len(self.ONI))) + list(range(0, dst +1))
                            new_link_info = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                            num_bends = self.get_num_bends(new_link_info, net_topology)
                            fai_total_crosstalk = self.get_fai_crosstalk_total(ONI_models_copy,current_wl,other_current_wl,new_link_info)
                            P_crosstalk = Ptx * (lp ** route_distance) * (lb ** num_bends) * fai_total_crosstalk
                            OP_crosstalk += P_crosstalk
                    else:
                        route_distance = dst - other_src if dst > other_src else len(self.ONI) - (other_src - dst)
                        new_path = list(range(other_src, dst + 1)) if other_src < dst else list(range(other_src, len(self.ONI))) + list(range(0, dst +1))
                        new_link_info = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
                        num_bends = self.get_num_bends(new_link_info, net_topology)
                        fai_total_crosstalk = self.get_fai_crosstalk_total(ONI_models_copy,current_wl,other_current_wl,new_link_info)
                        P_crosstalk = Ptx * (lp ** route_distance) * (lb ** num_bends) * fai_total_crosstalk
                        OP_crosstalk += P_crosstalk
                else:
                    P_crosstalk = 0
                    OP_crosstalk += P_crosstalk
            OSNR = 10 * np.log10(OP_signal / OP_crosstalk) + 17
            if OSNR > 65:
                OSNR = 65
            OSNRs[(src,dst)] = OSNR
        avg_OSNR = sum(OSNRs.values()) / len(OSNRs)
        return avg_OSNR, OSNRs
    
    #信息素更新
    def update_pheromone_assign(self,
                                pheromone: np.ndarray,
                                iteration: int,
                                best_osnrs_all_iters: List[float],
                                solutions_all_ants: List[Dict[Tuple[int,int],float]],
                                osnrs_all_ants: List[float],
                                glb_solution: Dict[Tuple[int,int],float],
                                glb_osnr: float) -> np.ndarray:
        #动态计算挥发率
        def get_dynamic_rho(iteration,rho_max=0.5,rho_min=0.1):
            base_rho = rho_max * np.exp(-3.0 * iteration / self.iterations)
            if iteration > 10 and (best_osnrs_all_iters[iteration] - best_osnrs_all_iters[iteration-10]) < 0.1:
                base_rho = min(rho_max, base_rho * 1.2)
            return max(rho_min, base_rho)
        current_rho = get_dynamic_rho(iteration)
        elite_boost = 2.5 - 2.0 * (iteration / self.iterations)
        #1.计算当前迭代所有蚂蚁OSNR的归一化权重
        osnr_min, osnr_max = np.min(osnrs_all_ants), np.max(osnrs_all_ants)
        weights = [(osnr - osnr_min) / (osnr_max - osnr_min + 1e-6) for osnr in osnrs_all_ants]
        #2.所有蚂蚁加权更新
        for idx, ant_solution in enumerate(solutions_all_ants):
            weight = weights[idx]
            for i in range(len(self.sorted_paths)):
                for j in range(len(self.com_wavelengths)):
                    #判断ant_solution中是否存在'self.sorted_paths[i]' : self.com_wavelengths[j]
                    if self.sorted_paths[i] in ant_solution and ant_solution[self.sorted_paths[i]] == self.com_wavelengths[j]:
                        delta_tao = self.Q / (self.K - osnrs_all_ants[idx]) * weight
                    else:
                        delta_tao = 0
                    pheromone[i][j] = pheromone[i][j]*(1-current_rho) + delta_tao
        #3.精英解强化
        for i in range(len(self.sorted_paths)):
            for j in range(len(self.com_wavelengths)):
                #判断ant_solutions中是否存在'self.sorted_paths[i]' : self.com_wavelengths[j]
                if self.sorted_paths[i] in glb_solution and glb_solution[self.sorted_paths[i]] == self.com_wavelengths[j]:
                    delta_tao = self.Q / (self.K - glb_osnr) * elite_boost
                else:
                    delta_tao = 0
                pheromone[i][j] = pheromone[i][j]*(1-current_rho) + delta_tao
        #4.信息素挥发与边界控制
        pheromone = np.clip(pheromone, self.tao_min, self.tao_max)
        return pheromone
    
    #更新信息素
    def update_pheromone_assign_tra(self,
                                   pheromone: np.ndarray,
                                   loc_solution: Dict[Tuple[int,int],float],
                                   loc_osnr: float) -> np.ndarray:
        for i in range(len(self.sorted_paths)):
            for j in range(len(self.com_wavelengths)):
                #判断current_iteration_best_solution中是否存在communication_pairs_info[i]' : communication_wavelengths[j]
                if self.sorted_paths[i] in loc_solution and loc_solution[self.sorted_paths[i]] == self.com_wavelengths[j]:
                    delta_tao = self.Q / (self.K - loc_osnr) 
                else:
                    delta_tao = 0
                pheromone[i][j] = (1 - self.rho) * pheromone[i][j] + delta_tao
        pheromone = np.clip(pheromone, self.tao_min, self.tao_max)
        return pheromone