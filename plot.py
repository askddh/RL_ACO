import matplotlib.pyplot as plt
from dataclasses import dataclass,field
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, TypeAlias
import copy

mapping_solution: TypeAlias = Dict[str, int]
assign_solution: TypeAlias = Dict[Tuple[int, int], float]
@dataclass
class data_container_g:
    solution: Optional[Union[mapping_solution, assign_solution]] = None
    osnr: float = -np.inf
    osnrs: List[Dict[Tuple[int, int], float]] = field(default_factory=list)
    best_osnrs_all_iters: List[float] = field(default_factory=list)
    conver_values: List[float] = field(default_factory=list)

    def global_update(self,data_container_l):
        if data_container_l.osnr > self.osnr:
            self.osnr = data_container_l.osnr
            self.solution = data_container_l.solution.copy()  # 保存全局最优解
            self.osnrs = data_container_l.osnrs
        #记录历次迭代最优解OSNR用于绘制概率图和收敛曲线图
        self.best_osnrs_all_iters.append(data_container_l.osnr)
        self.conver_values.append(self.osnr)

@dataclass
class data_container_l:
    solution: Optional[Union[mapping_solution, assign_solution]] = None
    osnr: float = -np.inf
    osnrs: List[float] = field(default_factory=list)
    solutions_all_ants: List[Dict[str, int]] = field(default_factory=list)
    osnrs_all_ants: List[float] = field(default_factory=list)
    
    def loc_update(self,ant_solution,avg_OSNR,OSNRs):
        self.solutions_all_ants.append(ant_solution)#更新信息素
        self.osnrs_all_ants.append(avg_OSNR)#更新信息素
        if avg_OSNR > self.osnr:#更新局部最优解
                self.osnr = avg_OSNR
                self.solution = ant_solution.copy()  # 保存当前最优解用于更新信息素
                self.osnrs = OSNRs

    
def statics_figure_mapping(iters,
                         osnrs_all_iters,
                         file_name='PhaseⅠ statics figure of all iterations under pip.svg'):
    base_path =r'd:\code\python\lianxi\RL_ACO_compare\ONoC_Optimization\results'  
    final_path = base_path + f'{file_name}'                              
    plt.figure(figsize=(10, 6))
    plt.scatter(range(iters), osnrs_all_iters, 
                    color='r',  
                    marker='o',  
                    s=1,  
                    label="PhaseⅠ statics figure of all iterations under pip")
    # plt.title('Statics figure of PhaseⅠ under pip')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('OSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def convergence_curve_figure_mapping(iters,
                            conver_values,
                            file_name='PhaseⅠ convergence curve under pip.svg'):
    base_path =r'd:\code\python\lianxi\RL_ACO_compare\ONoC_Optimization\results'  
    final_path = base_path + f'{file_name}'  
    plt.figure(figsize=(10,6))
    plt.plot(range(iters), conver_values,
            'r--',
            linewidth=1,
            label="Convergence curve of all iterations in PhaseⅠ under pip",
            marker='o',
            markersize=2)
    # plt.title('Convergence Curve of PhaseⅠ under pip')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('OSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(final_path,dpi=300, bbox_inches='tight')
    plt.close()   

def probs_figure_mapping(
                        osnrs_all_iters,
                        file_name='PhaseⅠ probability figure of all iterations under pip.svg'):
    base_path =r'd:\code\python\lianxi\RL_ACO_compare\ONoC_Optimization\results'  
    final_path = base_path + f'{file_name}'
    plt.figure(figsize=(10,6))
    # 先绘制密度直方图，获取数据
    n, bins, patches = plt.hist(
        osnrs_all_iters, 
        bins=100, 
        density=True,  # 先算密度
        color='blue', 
        alpha=0.7, 
        edgecolor='black'
    )
    
    # 关键：把密度换算成占比（密度×宽度=占比）
    bin_width = bins[1] - bins[0]  # 计算单个柱子的宽度
    for patch in patches:
        # 重新设置柱子高度=占比
        patch.set_height(patch.get_height() * bin_width)
    # plt.ylim(0,0.08)
    # 设置标签
    plt.xlabel('OSNR (dB)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close()
               



    