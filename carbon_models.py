import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gamma

# 一、土壤有机碳库模型

# （一）基础一阶动力学方程
def first_order_dynamics(t, x, k, A):
    """
    基础一阶动力学方程: dx/dt = -k*x + A
    
    参数:
        t: 时间
        x: 当前状态变量（有机碳含量）
        k: 衰减速率常数
        A: 碳输入速率
    返回:
        dx/dt: 状态变量变化率
    """
    return -k * x + A


# （二）不同结构分解模型

class D1Model:
    """D1模型（单库模型）"""
    def __init__(self, k=0.0231):
        self.k = k  # 衰减速率常数, 单位: y^-1
    
    def dynamics(self, t, x):
        """质量平衡方程: dx/dt = -k*x"""
        return -self.k * x
    
    def mean_residence_time(self):
        """平均滞留时间: tau = 1/k"""
        return 1 / self.k


class D2Model:
    """D2模型（串联双库模型）"""
    def __init__(self, r=0.870, k1=0.221, k2=0.0125):
        self.r = r    # 比例参数
        self.k1 = k1  # 库1衰减速率, y^-1
        self.k2 = k2  # 库2衰减速率, y^-1
    
    def dynamics(self, t, state, A):
        """
        质量平衡方程:
            dx1/dt = A - k1*x1
            dx2/dt = (1-r)*k1*x1 - k2*x2
        """
        x1, x2 = state
        dx1_dt = A - self.k1 * x1
        dx2_dt = (1 - self.r) * self.k1 * x1 - self.k2 * x2
        return [dx1_dt, dx2_dt]
    
    def mean_residence_time(self):
        """平均滞留时间: tau = 1/((1-r)*k1 + k2)"""
        return 1 / ((1 - self.r) * self.k1 + self.k2)


class D3Model:
    """D3模型（并联双库模型）"""
    def __init__(self, alpha=0.863, k1=0.221, k2=0.0125):
        self.alpha = alpha  # 分配比例
        self.k1 = k1        # 库1衰减速率, y^-1
        self.k2 = k2        # 库2衰减速率, y^-1
    
    def dynamics(self, t, state, A):
        """
        质量平衡方程:
            dx1/dt = alpha*A - k1*x1
            dx2/dt = (1-alpha)*A - k2*x2
        """
        x1, x2 = state
        dx1_dt = self.alpha * A - self.k1 * x1
        dx2_dt = (1 - self.alpha) * A - self.k2 * x2
        return [dx1_dt, dx2_dt]
    
    def mean_residence_time(self):
        """平均滞留时间: tau = alpha/k1 + (1-alpha)/k2"""
        return self.alpha / self.k1 + (1 - self.alpha) / self.k2


class D4Model:
    """D4模型（反馈模型）"""
    def __init__(self, r=0.879, k1=0.220, k2=0.0143):
        self.r = r    # 比例参数
        self.k1 = k1  # 库1衰减速率, y^-1
        self.k2 = k2  # 库2衰减速率, y^-1
    
    def dynamics(self, t, state, A):
        """
        质量平衡方程:
            dx1/dt = A - k1*x1
            dx2/dt = (1-r)*k1*x1 - k2*x2
        """
        x1, x2 = state
        dx1_dt = A - self.k1 * x1
        dx2_dt = (1 - self.r) * self.k1 * x1 - self.k2 * x2
        return [dx1_dt, dx2_dt]
    
    def mean_residence_time(self):
        """平均滞留时间: tau = 1/((1-r)*k1 + k2)"""
        return 1 / ((1 - self.r) * self.k1 + self.k2)


class L1aModel:
    """L1a模型 (Feng and Li, 2001)"""
    def __init__(self, a=0.236, b=0.0940, m=1.0):
        self.a = a  # 参数
        self.b = b  # 参数, y^-1
        self.m = m  # 参数
    
    def dynamics(self, t, x):
        """质量平衡方程: dx/dt = -(a + b*e^(-m*t))*x"""
        return -(self.a + self.b * np.exp(-self.m * t)) * x


class L2bModel:
    """L2b模型 (Rovira and Rovira, 2010)"""
    def __init__(self, k_base=0.02, t_opt=25, w_opt=0.6):
        self.k_base = k_base  # 基础衰减速率
        self.t_opt = t_opt    # 最适温度
        self.w_opt = w_opt    # 最适水分含量
    
    def temp_factor(self, T):
        """温度影响因子"""
        return np.exp(-0.5 * ((T - self.t_opt) / 10)**2)
    
    def water_factor(self, w):
        """水分影响因子"""
        return np.exp(-0.5 * ((w - self.w_opt) / 0.3)**2)
    
    def dynamics(self, t, x, T, w):
        """基于环境因子修正的衰减模型"""
        fT = self.temp_factor(T)
        fW = self.water_factor(w)
        return -self.k_base * fT * fW * x


class C1Model:
    """C1模型（Gamma分布模型, Bolker et al., 1998）"""
    def __init__(self, alpha=2.0, beta=0.5):
        self.alpha = alpha  # 形状参数
        self.beta = beta    # 尺度参数
    
    def probability_density(self, k):
        """概率密度函数: p(k) = (beta^alpha / Gamma(alpha)) * k^(alpha-1) * e^(-beta*k)"""
        return gamma.pdf(k, self.alpha, scale=1/self.beta)
    
    def dynamics(self, t, x):
        """质量平衡方程: dx/dt = -h * ∫p(k)x dk"""
        # 简化实现，使用Gamma分布的均值作为有效k值
        mean_k = self.alpha / self.beta  # Gamma分布的均值
        return -mean_k * x


class RothCModel:
    """Roth C模型"""
    def __init__(self):
        # 各碳库的分解速率常数 (y^-1)
        self.k_dpm = 10.0   # 易分解植物物质
        self.k_rpm = 0.3    # 难分解植物物质
        self.k_bio = 0.66   # 微生物生物量
        self.k_hum = 0.02   # 腐殖质
        self.k_iom = 0.0    # 惰性有机物质（不分解）
        
        # 周转时间 (年)
        self.tau_dpm = 1 / self.k_dpm
        self.tau_rpm = 1 / self.k_rpm
        self.tau_bio = 1 / self.k_bio
        self.tau_hum = 1 / self.k_hum
        self.tau_iom = float('inf')
    
    def dynamics(self, t, state):
        """
        各碳库的分解方程:
            d(DPM)/dt = -k_dpm * DPM
            d(RPM)/dt = -k_rpm * RPM
            d(BIO)/dt = -k_bio * BIO
            d(HUM)/dt = -k_hum * HUM
            d(IOM)/dt = 0
        """
        dpm, rpm, bio, hum, iom = state
        ddpm_dt = -self.k_dpm * dpm
        drpm_dt = -self.k_rpm * rpm
        dbio_dt = -self.k_bio * bio
        dhum_dt = -self.k_hum * hum
        diom_dt = -self.k_iom * iom  # 实际上为0
        return [ddpm_dt, drpm_dt, dbio_dt, dhum_dt, diom_dt]


class DSSATCenturyModel:
    """DSSAT模型的CENTURY模块"""
    def __init__(self):
        # 植物残体分解速率 (天^-1)
        self.k_ch = 0.2     # 碳水化合物
        self.k_cl = 0.05    # 纤维素
        self.k_ln = 0.0095  # 木质素
        self.k_som = 8.3e-5 # 土壤有机碳分解速率 (天^-1)
    
    def temp_factor(self, T):
        """温度因子 f(T)"""
        if T < 0:
            return 0.0
        elif 0 <= T <= 30:
            return T / 30.0
        else:  # T > 30
            return 1.0
    
    def water_factor(self, w, wfc):
        """水分因子 f(W)，基于土壤含水量与田间持水量的比值"""
        w_ratio = w / wfc
        if w_ratio < 0.2 or w_ratio > 1.2:
            return 0.2  # 过干或过湿时分解受抑制
        elif 0.2 <= w_ratio <= 0.8:
            return 0.5 + (w_ratio - 0.2) * 0.625
        else:  # 0.8 < w_ratio <= 1.2
            return 1.0 - (w_ratio - 0.8) * 2.0
    
    def cn_factor(self, cn_ratio):
        """碳氮比因子 f(C:N)"""
        if cn_ratio < 25:
            return 1.0
        else:
            return 25.0 / cn_ratio
    
    def residue_dynamics(self, t, state, T, w, wfc, cn_ratio):
        """
        植物残体分解方程:
            d(CH)/dt = -k_ch * CH * f(T) * f(W) * f(C:N)
            d(CL)/dt = -k_cl * CL * f(T) * f(W) * f(C:N)
            d(LN)/dt = -k_ln * LN * f(T) * f(W) * f(C:N)
        """
        ch, cl, ln = state
        fT = self.temp_factor(T)
        fW = self.water_factor(w, wfc)
        fCN = self.cn_factor(cn_ratio)
        
        dch_dt = -self.k_ch * ch * fT * fW * fCN
        dcl_dt = -self.k_cl * cl * fT * fW * fCN
        dln_dt = -self.k_ln * ln * fT * fW * fCN
        return [dch_dt, dcl_dt, dln_dt]
    
    def som_dynamics(self, t, som, T, w, wfc):
        """
        土壤有机碳分解方程:
            d(SOM)/dt = -k_som * SOM * f(T) * f(W)
        """
        fT = self.temp_factor(T)
        fW = self.water_factor(w, wfc)
        return -self.k_som * som * fT * fW