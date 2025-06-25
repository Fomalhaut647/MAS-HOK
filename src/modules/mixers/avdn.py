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
        
        # 注意力网络：从全局状态生成每个智能体的重要性权重
        # 网络结构: state_dim -> mixing_embed_dim -> n_agents
        self.attention_net = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, self.n_agents)
        )
        
        # 初始化网络参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络参数，使用Xavier uniform初始化"""
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, agent_qs, states):
        """
        前向传播
        
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
        
        # 应用softmax获得归一化的注意力权重，确保权重非负且和为1
        # 这满足了IGM（Individual-Global-Max）原则
        # shape: (batch_size * seq_len, n_agents)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 将注意力权重重塑回原始维度
        # shape: (batch_size, seq_len, n_agents)
        attention_weights = attention_weights.reshape(batch_size, seq_len, n_agents)
        
        # 将智能体Q值与对应的注意力权重相乘
        # shape: (batch_size, seq_len, n_agents)
        weighted_agent_qs = agent_qs * attention_weights
        
        # 在智能体维度上求和得到最终的团队Q值
        # shape: (batch_size, seq_len, 1)
        q_tot = th.sum(weighted_agent_qs, dim=2, keepdim=True)
        
        return q_tot 