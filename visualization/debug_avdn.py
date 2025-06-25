#!/usr/bin/env python3
"""
A-VDN调试脚本
用于分析注意力权重分布、训练稳定性和性能指标
"""

import torch as th
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
import sys

# 添加src到Python路径（如果直接运行此脚本）
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from src.modules.mixers.avdn import AVDNMixer

class AVDNDebugger:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.attention_weights_history = []
        
    def analyze_attention_weights(self, mixer, states, agent_qs):
        """分析注意力权重分布"""
        mixer.eval()
        with th.no_grad():
            batch_size, seq_len, n_agents = agent_qs.size()
            states_reshaped = states.reshape(-1, mixer.state_dim)
            
            # 获取注意力分数
            attention_scores = mixer.attention_net(states_reshaped)
            temperature = th.clamp(mixer.temperature, min=0.1, max=2.0)
            attention_scores = attention_scores / temperature
            attention_weights = th.softmax(attention_scores, dim=-1)
            
            # 计算统计信息
            weights_mean = attention_weights.mean(dim=0)
            weights_std = attention_weights.std(dim=0)
            weights_entropy = -th.sum(attention_weights * th.log(attention_weights + 1e-8), dim=-1).mean()
            
            # 计算残差权重
            residual_weight = th.sigmoid(mixer.residual_weight).item()
            temperature_val = temperature.item()
            
            return {
                'weights_mean': weights_mean.cpu().numpy(),
                'weights_std': weights_std.cpu().numpy(),
                'entropy': weights_entropy.item(),
                'residual_weight': residual_weight,
                'temperature': temperature_val,
                'raw_weights': attention_weights.cpu().numpy()
            }
    
    def plot_attention_analysis(self, analysis_results):
        """可视化注意力分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 平均注意力权重
        axes[0, 0].bar(range(len(analysis_results['weights_mean'])), 
                       analysis_results['weights_mean'])
        axes[0, 0].set_title(f'Average Attention Weights (Entropy: {analysis_results["entropy"]:.3f})')
        axes[0, 0].set_xlabel('Agent ID')
        axes[0, 0].set_ylabel('Average Weight')
        
        # 2. 权重标准差
        axes[0, 1].bar(range(len(analysis_results['weights_std'])), 
                       analysis_results['weights_std'])
        axes[0, 1].set_title('Attention Weight Standard Deviation')
        axes[0, 1].set_xlabel('Agent ID')
        axes[0, 1].set_ylabel('Standard Deviation')
        
        # 3. 权重分布热图
        weights = analysis_results['raw_weights'][:100]  # 显示前100个样本
        im = axes[1, 0].imshow(weights.T, aspect='auto', cmap='Blues')
        axes[1, 0].set_title('Attention Weight Distribution Heatmap')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Agent ID')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. 关键参数
        params_text = f"""Key Parameters:
Temperature: {analysis_results['temperature']:.3f}
Residual Weight: {analysis_results['residual_weight']:.3f}
Attention Entropy: {analysis_results['entropy']:.3f}

Weight Statistics:
Max Weight: {analysis_results['weights_mean'].max():.3f}
Min Weight: {analysis_results['weights_mean'].min():.3f}
Weight Variance: {analysis_results['weights_mean'].var():.3f}
"""
        axes[1, 1].text(0.1, 0.5, params_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Parameter Summary')
        
        plt.tight_layout()
        return fig
    
    def compare_with_vdn(self, mixer, states, agent_qs):
        """对比A-VDN和VDN的输出"""
        mixer.eval()
        with th.no_grad():
            # A-VDN输出
            avdn_output = mixer(agent_qs, states)
            
            # VDN输出（简单加法）
            vdn_output = th.sum(agent_qs, dim=2, keepdim=True)
            
            # 计算差异
            diff = avdn_output - vdn_output
            diff_mean = diff.mean().item()
            diff_std = diff.std().item()
            
            return {
                'avdn_mean': avdn_output.mean().item(),
                'vdn_mean': vdn_output.mean().item(),
                'diff_mean': diff_mean,
                'diff_std': diff_std,
                'correlation': th.corrcoef(th.stack([avdn_output.flatten(), 
                                                   vdn_output.flatten()]))[0,1].item()
            }
    
    def load_training_metrics(self, results_path):
        """加载训练指标"""
        info_path = os.path.join(results_path, 'info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            return info
        return None
    
    def analyze_training_stability(self, info):
        """分析训练稳定性"""
        if not info:
            return None
            
        loss_td = info.get('loss_td', [])
        if not loss_td:
            return None
            
        # 计算损失统计
        loss_mean = np.mean(loss_td)
        loss_std = np.std(loss_td)
        loss_trend = np.polyfit(range(len(loss_td)), loss_td, 1)[0]  # 线性趋势
        
        # 检测异常值（超过3个标准差）
        outliers = [x for x in loss_td if abs(x - loss_mean) > 3 * loss_std]
        
        return {
            'loss_mean': loss_mean,
            'loss_std': loss_std,
            'loss_trend': loss_trend,
            'outlier_count': len(outliers),
            'outlier_ratio': len(outliers) / len(loss_td),
            'final_loss': loss_td[-10:] if len(loss_td) >= 10 else loss_td  # 最后10个值
        }

def main():
    """主函数 - 运行A-VDN诊断"""
    print("🔍 A-VDN诊断工具")
    print("=" * 50)
    
    # 创建调试器
    debugger = AVDNDebugger()
    
    # 创建模拟数据进行测试
    n_agents = 5
    state_dim = 100
    batch_size = 32
    seq_len = 10
    
    # 模拟args对象
    class Args:
        n_agents = 5
        state_shape = (100,)
        mixing_embed_dim = 32
    
    args = Args()
    
    # 创建A-VDN mixer
    mixer = AVDNMixer(args)
    
    # 生成模拟数据
    agent_qs = th.randn(batch_size, seq_len, n_agents)
    states = th.randn(batch_size, seq_len, state_dim)
    
    # 分析注意力权重
    print("📊 分析注意力权重分布...")
    analysis = debugger.analyze_attention_weights(mixer, states, agent_qs)
    
    # 可视化结果
    fig = debugger.plot_attention_analysis(analysis)
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'avdn_attention_analysis.png'), dpi=300, bbox_inches='tight')
    
    print("✅ 注意力分析图已保存为: avdn_attention_analysis.png")
    
    # 对比A-VDN和VDN
    print("\n🔄 对比A-VDN和VDN输出...")
    comparison = debugger.compare_with_vdn(mixer, states, agent_qs)
    print(f"A-VDN平均输出: {comparison['avdn_mean']:.4f}")
    print(f"VDN平均输出: {comparison['vdn_mean']:.4f}")
    print(f"输出差异均值: {comparison['diff_mean']:.4f}")
    print(f"输出差异标准差: {comparison['diff_std']:.4f}")
    print(f"相关性: {comparison['correlation']:.4f}")
    
    print("\n✨ 诊断完成！请查看生成的图表和输出结果。")

if __name__ == "__main__":
    main() 