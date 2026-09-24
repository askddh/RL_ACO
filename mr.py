"""
问题描述：
功率衰减系数为0,端口输出的传输效率过于离谱,要么大于1,要么量级差的太大,明显不正常,甚至有时出现负值。
问题反思：
1.对于公式中的参数要搞清楚参数的单位问题。
2.公式理解透彻后再进行编程实现。
    #功率衰减系数的单位问题、线性值转换问题；
    #相移公式弧度转换问题；
    #当光信号注入微环时,不同波长光信号的折射率sellmeier色散问题;
     从网上找了微环仿真代码,通过matlab仿真微环的工作原理,了解微环的工作原理。
     耗时2天,后来发现工作量多余了,不需要考虑这些问题。
"""
import math
class MR:
    def __init__(self, λ_mr, state=0):
        self.r1 = 0.988                             #微环MR自耦合系数
        self.r2 = 0.988    
        self.k1 = math.sqrt(1 - self.r1**2)         #微环MR交叉耦合系数
        self.k2 = math.sqrt(1 - self.r2**2)
        self.R = 1600                               #微环MR半径，单位nm
        self.mode = 17                              #微环MR模式
        self.λ_mr = λ_mr                            #微环MR工作波长，单位nm
        self.state = state                          #微环MR状态，0表示OFF，1表示ON
        self.pi = math.pi                           #圆周率
        self.deltaλ = 2                           #微环MR在ON和OFF状态之间的波长漂移，单位nm
        self.alpha = 200                            #微环MR的功率衰减系数，单位dB/m
        self.FSR = 59                               #FSR,单位nm
        self.L = 2 * self.R * self.pi * 1e-9        #微环MR的长度，单位m
    #计算微环的谐振波长
    @property
    def λ_res(self):
        return self.λ_mr if self.state == 1 else self.λ_mr + self.deltaλ
    #计算微环MR的有效折射率
    @property
    def n_res(self):
        return (self.mode * self.λ_res * 1e-9) / (self.L)
    #计算群折射率n_group
    @property
    def n_group(self):
        return (self.λ_res * 1e-9) ** 2 / (self.L * self.FSR)
    #计算微环的信号传输幅度a
    @property
    def a(self):
        return math.sqrt(math.exp(-self.alpha * self.L))  
    #计算FSR
    @property
    def fsr(self):
        return ((self.λ_res * 1e-9) ** 2 ) / (self.L * self.n_group)
    #计算当前输入信号相移
    def theta(self,λ_signal):
        return (2 * self.pi * self.mode * self.λ_res * 1e-9) / (λ_signal * 1e-9)
    #计算through端透射率
    def fai_t(self , λ_signal , xita):
        return ((self.a ** 2 *self.r2 ** 2) - 2 * self.a *  self.r1 * self.r2 * math.cos(xita) + self.r1 ** 2) / (1 - 2 * self.a  * self.r1 * self.r2 * math.cos(xita) + (self.a *self.r1 * self.r2) ** 2 )
    #计算drop端透射率
    def fai_d(self, λ_signal, xita):
        return (self.a  * (self.k1 ** 2 ) * (self.k2 ** 2)) / (1- 2 * self.a * self.r1 * self.r2 * math.cos(xita) + (self.a * self.r1 * self.r2) **2)


if __name__ == '__main__':
    mr1 = MR(λ_mr=1550, state=1)  
    print(f'信号谐振波长λ_res={mr1.λ_res}nm')
    print(f'有效折射率n_res={mr1.n_res}')  
    print(f'信号传输幅度a={mr1.a}') 
    print(f'FSR={mr1.fsr}nm')  
    lambda_signal = 1550
    print(f'当前输入信号波长λ_signal={lambda_signal}nm')
    print(f'当前输入信号波长与谐振波长相位差={mr1.theta(lambda_signal)}')   
    xita = mr1.theta(lambda_signal)
    print(f'through端信号传输效率={mr1.fai_t(lambda_signal,xita)}')
    print(f'drop端信号传输效率={mr1.fai_d(lambda_signal,xita)}')