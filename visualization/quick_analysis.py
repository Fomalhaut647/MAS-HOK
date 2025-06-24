#!/usr/bin/env python3
"""
Quick Analysis of VDN Algorithm Performance in HOK Environment
"""

import ast
import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze():
    # Load data
    with open('results/sacred/hok/monster_last_hp/2025-06-24 02:41_monster_lasthp_vdn_env.txt', 'r') as f:
        hp_data = np.array(ast.literal_eval(f.read().strip()))
    
    initial_hp = 30000
    damage_dealt = initial_hp - hp_data
    damage_ratio = damage_dealt / initial_hp * 100
    
    # Basic statistics
    print("="*80)
    print("🎮 VDN 算法在 HOK 环境中的评估结果分析")
    print("="*80)
    print(f"📊 评估总数: {len(hp_data):,} 次")
    print(f"🏆 完全击败怪物次数: {np.sum(hp_data == 0):,} 次")
    print(f"📈 胜利率: {np.sum(hp_data == 0) / len(hp_data) * 100:.2f}%")
    print()
    print("💪 伤害效果分析:")
    print(f"   平均剩余血量: {np.mean(hp_data):.0f} / {initial_hp}")
    print(f"   平均造成伤害: {np.mean(damage_dealt):.0f}")
    print(f"   平均伤害率: {np.mean(damage_ratio):.2f}%")
    print(f"   最佳表现: 仅剩 {np.min(hp_data):,} 血量")
    print(f"   最差表现: 剩余 {np.max(hp_data):,} 血量 (未造成伤害)")
    print()
    print("📊 性能稳定性:")
    print(f"   伤害标准差: {np.std(damage_dealt):.0f}")
    print(f"   变异系数: {np.std(damage_dealt)/np.mean(damage_dealt)*100:.1f}%")
    print()
    
    # Performance insights
    print("🔍 关键洞察:")
    
    # 1. 零伤害次数
    zero_damage = np.sum(hp_data == initial_hp)
    if zero_damage > 0:
        print(f"   ⚠️  有 {zero_damage} 次 ({zero_damage/len(hp_data)*100:.1f}%) 完全未造成伤害")
    
    # 2. 高伤害次数（>80%）
    high_damage = np.sum(damage_ratio > 80)
    print(f"   🎯 造成 >80% 伤害的次数: {high_damage} ({high_damage/len(hp_data)*100:.1f}%)")
    
    # 3. 低伤害次数（<20%）
    low_damage = np.sum(damage_ratio < 20)
    print(f"   🔴 造成 <20% 伤害的次数: {low_damage} ({low_damage/len(hp_data)*100:.1f}%)")
    
    # 4. 性能趋势
    window = 1000
    if len(hp_data) >= window:
        early_avg = np.mean(damage_ratio[:window])
        late_avg = np.mean(damage_ratio[-window:])
        trend = late_avg - early_avg
        
        print(f"   📈 学习趋势: 前{window}轮 vs 后{window}轮: {trend:+.2f}% 变化")
        if trend > 5:
            print("      ✅ 显著改善")
        elif trend > 0:
            print("      🟡 轻微改善")
        elif trend > -5:
            print("      🟡 基本稳定")
        else:
            print("      🔴 性能下降")
    
    print("="*80)
    
    # Create simple visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. HP over time
    ax1.plot(hp_data, alpha=0.7, color='red', linewidth=0.5)
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    ax1.set_title('Monster HP Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Remaining HP')
    ax1.grid(True, alpha=0.3)
    
    # 2. Damage distribution
    ax2.hist(damage_dealt, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=np.mean(damage_dealt), color='red', linestyle='--', 
                label=f'Mean: {np.mean(damage_dealt):.0f}')
    ax2.set_title('Damage Distribution')
    ax2.set_xlabel('Damage Dealt')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Moving average
    if len(hp_data) >= 100:
        moving_avg = np.convolve(damage_dealt, np.ones(100)/100, mode='valid')
        ax3.plot(range(99, len(hp_data)), moving_avg, color='purple', linewidth=2)
        ax3.set_title('Damage Trend (100-episode moving average)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Damage')
        ax3.grid(True, alpha=0.3)
    
    # 4. Performance quantiles
    quartiles = np.percentile(damage_ratio, [25, 50, 75])
    colors = ['red', 'orange', 'yellow', 'green']
    ranges = [0, 25, 50, 75, 100]
    counts = []
    
    for i in range(len(ranges)-1):
        count = np.sum((damage_ratio >= ranges[i]) & (damage_ratio < ranges[i+1]))
        counts.append(count)
    
    ax4.bar(range(len(counts)), counts, color=colors, alpha=0.7)
    ax4.set_title('Performance Distribution')
    ax4.set_xlabel('Damage Range (%)')
    ax4.set_ylabel('Episode Count')
    ax4.set_xticks(range(len(counts)))
    ax4.set_xticklabels(['0-25%', '25-50%', '50-75%', '75-100%'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization_results/quick_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 可视化图表已保存到: visualization_results/quick_analysis.png")
    
    return hp_data, damage_dealt, damage_ratio

if __name__ == "__main__":
    load_and_analyze() 