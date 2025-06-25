import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AVDNMixer(nn.Module):
    """
    Attention-based Value-Decomposition Network (A-VDN) Mixer
    
    A-VDN改进了传统VDN的简单加法分解假设，通过引入注意力机制来为每个智能体
    的Q值分配动态权重。这使得算法能够根据全局状态来判断不同智能体在当前情况下
    的重要性，从而更好地建模智能体间的协同关系。
    
    核心公式: Q_tot(τ, u, s) = Σ_i α_i(s) * Q_i(τ_i, u_i)
    其中 α_i(s) 是基于全局状态s计算的注意力权重
    """
    
    def __init__(self, args):
        super(AVDNMixer, self).__init__()
        
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.mixing_embed_dim = getattr(args, 'mixing_embed_dim', 32)
        
        # 使用更保守的网络结构提高训练稳定性
        self.attention_net = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout防止过拟合
            nn.Linear(self.mixing_embed_dim, self.n_agents)
        )
        
        # 可学习的温度参数，控制注意力分布的尖锐程度
        self.temperature = nn.Parameter(th.ones(1))
        
        # 残差连接权重，平衡注意力机制和简单加法
        self.residual_weight = nn.Parameter(th.tensor(0.5))
        
        # 初始化网络参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络参数，使用更保守的初始化方法"""
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                # 使用更小的初始化范围，提高训练稳定性
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, agent_qs, states):
        """
        前向传播 - 改进版本，提高训练稳定性
        
        Args:
            agent_qs: 智能体Q值, shape: (batch_size, seq_len, n_agents)
            states: 全局状态, shape: (batch_size, seq_len, state_dim)
            
        Returns:
            q_tot: 团队总Q值, shape: (batch_size, seq_len, 1)
        """
        batch_size, seq_len, n_agents = agent_qs.size()
        
        # 将states重塑为二维以便输入注意力网络
        # shape: (batch_size * seq_len, state_dim)
        states_reshaped = states.reshape(-1, self.state_dim)
        
        # 通过注意力网络生成每个智能体的分数
        # shape: (batch_size * seq_len, n_agents)
        attention_scores = self.attention_net(states_reshaped)
        
        # 使用温度参数调节注意力分布的尖锐程度
        # 温度越高，分布越平滑；温度越低，分布越集中
        temperature = th.clamp(self.temperature, min=0.1, max=2.0)  # 限制温度范围
        attention_scores = attention_scores / temperature
        
        # 应用softmax获得归一化的注意力权重
        # shape: (batch_size * seq_len, n_agents)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 将注意力权重重塑回原始维度
        # shape: (batch_size, seq_len, n_agents)
        attention_weights = attention_weights.reshape(batch_size, seq_len, n_agents)
        
        # 计算注意力加权的Q值
        # shape: (batch_size, seq_len, n_agents)
        weighted_agent_qs = agent_qs * attention_weights
        attention_q_tot = th.sum(weighted_agent_qs, dim=2, keepdim=True)
        
        # 计算简单加法的Q值（类似VDN）
        vdn_q_tot = th.sum(agent_qs, dim=2, keepdim=True)
        
        # 使用残差连接结合两种方法，提高训练稳定性
        residual_weight = th.sigmoid(self.residual_weight)  # 确保权重在[0,1]之间
        q_tot = residual_weight * attention_q_tot + (1 - residual_weight) * vdn_q_tot
        
        return q_tot 