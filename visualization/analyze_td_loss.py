#!/usr/bin/env python3
"""
TD损失稳定性分析工具
用于对比A-VDN和VDN的训练稳定性
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_loss_data(info_json_path):
    """从info.json文件中加载TD损失数据"""
    try:
        with open(info_json_path, 'r') as f:
            info = json.load(f)
        return info.get('loss_td', [])
    except Exception as e:
        print(f"加载文件失败 {info_json_path}: {e}")
        return []

def analyze_stability(loss_data, name):
    """分析损失稳定性"""
    if not loss_data:
        return None
    
    loss_array = np.array(loss_data)
    
    # 基本统计
    mean_loss = np.mean(loss_array)
    std_loss = np.std(loss_array)
    min_loss = np.min(loss_array)
    max_loss = np.max(loss_array)
    
    # 变异系数 (Coefficient of Variation)
    cv = std_loss / mean_loss if mean_loss > 0 else float('inf')
    
    # 趋势分析 (线性回归斜率)
    x = np.arange(len(loss_array))
    trend = np.polyfit(x, loss_array, 1)[0]
    
    # 波动性分析
    diff = np.diff(loss_array)
    volatility = np.std(diff)
    
    # 异常值检测 (超过3个标准差)
    outliers = loss_array[np.abs(loss_array - mean_loss) > 3 * std_loss]
    outlier_ratio = len(outliers) / len(loss_array)
    
    # 稳定性评分 (越低越稳定)
    stability_score = cv + volatility/mean_loss + outlier_ratio
    
    return {
        'name': name,
        'count': len(loss_array),
        'mean': mean_loss,
        'std': std_loss,
        'min': min_loss,
        'max': max_loss,
        'cv': cv,
        'trend': trend,
        'volatility': volatility,
        'outlier_count': len(outliers),
        'outlier_ratio': outlier_ratio,
        'stability_score': stability_score,
        'data': loss_array
    }

def plot_comparison(avdn_stats, vdn_stats):
    """绘制对比分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 损失曲线对比
    axes[0, 0].plot(avdn_stats['data'], label='A-VDN', alpha=0.7, color='red')
    axes[0, 0].plot(vdn_stats['data'], label='VDN', alpha=0.7, color='blue')
    axes[0, 0].set_title('TD Loss Comparison')
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('TD Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 统计对比
    metrics = ['mean', 'std', 'cv', 'volatility', 'outlier_ratio']
    avdn_values = [avdn_stats[m] for m in metrics]
    vdn_values = [vdn_stats[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, avdn_values, width, label='A-VDN', color='red', alpha=0.7)
    axes[0, 1].bar(x + width/2, vdn_values, width, label='VDN', color='blue', alpha=0.7)
    axes[0, 1].set_title('Stability Metrics Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Mean', 'Std Dev', 'CV', 'Volatility', 'Outlier Ratio'])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 滑动窗口方差
    window_size = 50
    avdn_rolling_var = []
    vdn_rolling_var = []
    
    for i in range(window_size, len(avdn_stats['data'])):
        avdn_rolling_var.append(np.var(avdn_stats['data'][i-window_size:i]))
    
    for i in range(window_size, len(vdn_stats['data'])):
        vdn_rolling_var.append(np.var(vdn_stats['data'][i-window_size:i]))
    
    axes[1, 0].plot(avdn_rolling_var, label='A-VDN', color='red', alpha=0.7)
    axes[1, 0].plot(vdn_rolling_var, label='VDN', color='blue', alpha=0.7)
    axes[1, 0].set_title(f'Rolling Window Variance (Window Size={window_size})')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计总结表格
    summary_text = f"""
Stability Analysis Comparison
{'='*40}

A-VDN:
- Average Loss: {avdn_stats['mean']:.2f}
- Standard Deviation: {avdn_stats['std']:.2f}
- Coefficient of Variation (CV): {avdn_stats['cv']:.3f}
- Max Loss: {avdn_stats['max']:.2f}
- Min Loss: {avdn_stats['min']:.2f}
- Outlier Ratio: {avdn_stats['outlier_ratio']:.1%}
- Stability Score: {avdn_stats['stability_score']:.3f}

VDN:
- Average Loss: {vdn_stats['mean']:.2f}
- Standard Deviation: {vdn_stats['std']:.2f}
- Coefficient of Variation (CV): {vdn_stats['cv']:.3f}
- Max Loss: {vdn_stats['max']:.2f}
- Min Loss: {vdn_stats['min']:.2f}
- Outlier Ratio: {vdn_stats['outlier_ratio']:.1%}
- Stability Score: {vdn_stats['stability_score']:.3f}

Conclusion:
VDN is {avdn_stats['stability_score']/vdn_stats['stability_score']:.1f} times more stable than A-VDN
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("🔍 TD损失稳定性分析工具")
    print("=" * 50)
    
    # 定义文件路径
    avdn_path = "results/sacred/hok/avdn_env/1/info.json"
    vdn_path = "results/sacred/hok/vdn_env/3/info.json"
    
    # 检查文件是否存在
    if not os.path.exists(avdn_path):
        print(f"❌ A-VDN结果文件不存在: {avdn_path}")
        return
    
    if not os.path.exists(vdn_path):
        print(f"❌ VDN结果文件不存在: {vdn_path}")
        return
    
    # 加载数据
    print("📈 加载训练数据...")
    avdn_loss = load_loss_data(avdn_path)
    vdn_loss = load_loss_data(vdn_path)
    
    if not avdn_loss or not vdn_loss:
        print("❌ 无法加载损失数据")
        return
    
    # 分析稳定性
    print("🔬 分析稳定性...")
    avdn_stats = analyze_stability(avdn_loss, "A-VDN")
    vdn_stats = analyze_stability(vdn_loss, "VDN")
    
    if avdn_stats is None or vdn_stats is None:
        print("❌ 稳定性分析失败，请检查损失数据。")
        return
    
    # 打印结果
    print(f"\n📊 分析结果:")
    print(f"A-VDN: 平均损失={avdn_stats['mean']:.2f}, 标准差={avdn_stats['std']:.2f}, 变异系数={avdn_stats['cv']:.3f}")
    print(f"VDN:   平均损失={vdn_stats['mean']:.2f}, 标准差={vdn_stats['std']:.2f}, 变异系数={vdn_stats['cv']:.3f}")
    print(f"\n🎯 稳定性评分 (越低越稳定):")
    print(f"A-VDN: {avdn_stats['stability_score']:.3f}")
    print(f"VDN:   {vdn_stats['stability_score']:.3f}")
    print(f"VDN比A-VDN稳定 {avdn_stats['stability_score']/vdn_stats['stability_score']:.1f}倍")
    
    # 绘制对比图
    print("\n📊 生成对比图...")
    fig = plot_comparison(avdn_stats, vdn_stats)
    
    # 保存图片
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "td_loss_stability_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 分析图已保存: {output_file}")
    
    # 显示关键发现
    print(f"\n🔍 关键发现:")
    print(f"1. A-VDN的损失范围: {avdn_stats['min']:.1f} - {avdn_stats['max']:.1f}")
    print(f"2. VDN的损失范围: {vdn_stats['min']:.1f} - {vdn_stats['max']:.1f}")
    print(f"3. A-VDN的异常值比例: {avdn_stats['outlier_ratio']:.1%}")
    print(f"4. VDN的异常值比例: {vdn_stats['outlier_ratio']:.1%}")
    
    if avdn_stats['max'] > 90:
        print("⚠️  A-VDN出现了超过90的极高损失值，表明训练极不稳定！")
    
    if avdn_stats['cv'] > vdn_stats['cv'] * 2:
        print("⚠️  A-VDN的变异系数显著高于VDN，训练波动性很大！")

if __name__ == "__main__":
    main() 