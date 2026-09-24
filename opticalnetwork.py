from typing import List, Tuple, Dict, Union, Any, Optional, Set
import numpy as np
import random
import math
from . import MR
class OpticalNetworkEnv:
    def __init__(self,C,communication):
        self.C = C
        self.communication = communication
        self.ONI = list(range(len(C)))
        self.N_wl = len(self.ONI)
        self.com_wavelengths = [1550 + i * (59 / self.N_wl) for i in range(self.N_wl)]
        self.W = self.get_w()
        self.dis_matrix = self.get_distance_matrix()
        self.ONI_models = self.optical_network_interface()

    #通信权重矩阵W
    def get_w(self) -> np.ndarray:
        num_cores = len(self.C)
        W = np.zeros((num_cores, num_cores))
        for src, targets in self.communication.items():
            src_index = self.C.index(src)                
            for dest, weight in targets:
                dest_index = self.C.index(dest)                    
                W[src_index][dest_index] = weight
        return W
    
    #距离矩阵
    def get_distance_matrix(self) -> np.ndarray:
        dis_matrix = np.zeros((len(self.ONI),len(self.ONI)))
        for i in range(len(self.ONI)):  
            for j in range(len(self.ONI)):
                if i < j:
                    dis_matrix[i][j] = j - i
                    dis_matrix[j][i] = len(self.ONI) - (j - i)
                elif i == j:
                    dis_matrix[i][j] = 0
        return dis_matrix
    
    #光网络接口建模
    def optical_network_interface(self) -> Dict[int, List[MR]]:
        ONI_models = {}  #ONI_models = {0: [mr1,mr2,mr3,mr4,mr5,mr6,mr7,mr8], 1: [mr1,mr2,mr3,mr4,mr5,mr6,mr7,mr8],...}
        #初始化每个节点内各个微环的谐振波长（默认off state）
        for i in range(len(self.ONI)):
            ONI_models[i] = []
            for j in range(len(self.com_wavelengths)):
                mrj = MR(self.com_wavelengths[j])
                ONI_models[i].append(mrj)
        return ONI_models