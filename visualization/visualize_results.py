#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Reinforcement Learning Evaluation Results Visualization Tool
For analyzing VDN algorithm performance in HOK environment
"""

import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_hp_data(file_path):
    """Load monster remaining HP data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        hp_data = ast.literal_eval(content)
    return np.array(hp_data)

def analyze_performance(hp_data, initial_hp=30000):
    """Analyze performance metrics"""
    damage_dealt = initial_hp - hp_data
    damage_ratio = damage_dealt / initial_hp * 100
    
    stats = {
        'total_evaluations': len(hp_data),
        'complete_defeats': np.sum(hp_data == 0),
        'win_rate': np.sum(hp_data == 0) / len(hp_data) * 100,
        'avg_remaining_hp': np.mean(hp_data),
        'avg_damage': np.mean(damage_dealt),
        'avg_damage_ratio': np.mean(damage_ratio),
        'min_remaining_hp': np.min(hp_data),
        'max_remaining_hp': np.max(hp_data),
        'damage_std': np.std(damage_dealt)
    }
    
    return stats, damage_dealt, damage_ratio

def create_visualizations(hp_data, damage_dealt, damage_ratio, stats, save_dir='./visualization_results'):
    """Create multiple visualization charts"""
    Path(save_dir).mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. HP time series
    plt.subplot(3, 3, 1)
    plt.plot(hp_data, alpha=0.8, linewidth=1, color='red')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Complete Defeat Line')
    plt.title('Monster Remaining HP Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Remaining HP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. HP distribution
    plt.subplot(3, 3, 2)
    plt.hist(hp_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=stats['avg_remaining_hp'], color='red', linestyle='--', 
                label=f'Mean: {stats["avg_remaining_hp"]:.0f}')
    plt.title('HP Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Remaining HP')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Damage time series
    plt.subplot(3, 3, 3)
    plt.plot(damage_dealt, alpha=0.8, linewidth=1, color='blue')
    plt.axhline(y=30000, color='green', linestyle='--', alpha=0.7, label='Complete Defeat Line')
    plt.title('Damage Dealt Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Damage Dealt')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Damage ratio distribution
    plt.subplot(3, 3, 4)
    plt.hist(damage_ratio, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=stats['avg_damage_ratio'], color='red', linestyle='--', 
                label=f'Mean: {stats["avg_damage_ratio"]:.1f}%')
    plt.title('Damage Ratio Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Damage Ratio (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Moving average performance
    window_size = 100
    if len(hp_data) >= window_size:
        plt.subplot(3, 3, 5)
        moving_avg_hp = np.convolve(hp_data, np.ones(window_size)/window_size, mode='valid')
        moving_avg_damage = np.convolve(damage_dealt, np.ones(window_size)/window_size, mode='valid')
        
        x_range = range(window_size-1, len(hp_data))
        plt.plot(x_range, moving_avg_hp, label=f'HP (MA-{window_size})', color='red')
        plt.plot(x_range, moving_avg_damage, label=f'Damage (MA-{window_size})', color='blue')
        plt.title(f'Performance Trend (Moving Average: {window_size})', fontsize=14, fontweight='bold')
        plt.xlabel('Evaluation Episode')
        plt.ylabel('HP/Damage')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 6. Cumulative win rate
    plt.subplot(3, 3, 6)
    wins = (hp_data == 0).astype(int)
    cumulative_win_rate = np.cumsum(wins) / np.arange(1, len(wins) + 1) * 100
    plt.plot(cumulative_win_rate, color='green', linewidth=2)
    plt.title('Cumulative Win Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Cumulative Win Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # 7. Segmented performance
    plt.subplot(3, 3, 7)
    segments = 10
    segment_size = len(hp_data) // segments
    segment_stats = []
    
    for i in range(segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < segments - 1 else len(hp_data)
        segment_data = hp_data[start_idx:end_idx]
        segment_win_rate = np.sum(segment_data == 0) / len(segment_data) * 100
        segment_stats.append(segment_win_rate)
    
    plt.bar(range(1, segments + 1), segment_stats, alpha=0.7, color='orange')
    plt.title('Segmented Win Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Training Phase')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # 8. Box plot
    plt.subplot(3, 3, 8)
    box_data = [hp_data, damage_dealt]
    plt.boxplot(box_data)
    plt.xticks([1, 2], ['Remaining HP', 'Damage Dealt'])
    plt.title('Data Distribution Box Plot', fontsize=14, fontweight='bold')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3)
    
    # 9. Performance summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    Performance Summary Report
    
    Total Evaluations: {stats['total_evaluations']:,}
    Complete Defeats: {stats['complete_defeats']:,}
    Win Rate: {stats['win_rate']:.2f}%
    
    Average Remaining HP: {stats['avg_remaining_hp']:.0f}
    Average Damage: {stats['avg_damage']:.0f}
    Average Damage Ratio: {stats['avg_damage_ratio']:.2f}%
    
    Best Performance: {stats['min_remaining_hp']:.0f} HP remaining
    Worst Performance: {stats['max_remaining_hp']:.0f} HP remaining
    Performance Stability: {stats['damage_std']:.0f} (damage std)
    """
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/eval_visual.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_dir}/eval_visual.png")
    
    return fig

def print_detailed_stats(stats):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("    VDN Algorithm HOK Environment Evaluation Report")
    print("="*60)
    print(f"📊 Total Evaluations: {stats['total_evaluations']:,}")
    print(f"🏆 Complete Defeats: {stats['complete_defeats']:,}")
    print(f"📈 Win Rate: {stats['win_rate']:.2f}%")
    print(f"❤️  Average Remaining HP: {stats['avg_remaining_hp']:.0f}")
    print(f"⚔️  Average Damage: {stats['avg_damage']:.0f}")
    print(f"💪 Average Damage Ratio: {stats['avg_damage_ratio']:.2f}%")
    print(f"🎯 Best Performance: {stats['min_remaining_hp']:.0f} HP remaining")
    print(f"😅 Worst Performance: {stats['max_remaining_hp']:.0f} HP remaining")
    print(f"📊 Performance Stability: {stats['damage_std']:.0f} (damage std)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Visualize MARL evaluation results')
    parser.add_argument('--file', '-f', 
                        default='results/sacred/hok/monster_last_hp/2025-06-24 02:41_monster_lasthp_vdn_env.txt',
                        help='Evaluation results file path')
    parser.add_argument('--output', '-o', default='./visualization_results',
                        help='Output directory')
    parser.add_argument('--show', action='store_true',
                        help='Display charts')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading evaluation results data...")
        hp_data = load_hp_data(args.file)
        print(f"Successfully loaded {len(hp_data)} evaluation records")
        
        # Analyze performance
        print("Analyzing performance metrics...")
        stats, damage_dealt, damage_ratio = analyze_performance(hp_data)
        
        # Print statistics
        print_detailed_stats(stats)
        
        # Create visualizations
        print("Generating visualization charts...")
        fig = create_visualizations(hp_data, damage_dealt, damage_ratio, stats, args.output)
        
        # Show charts
        if args.show:
            plt.show()
        
        print(f"\n✅ Visualization completed! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check file path and format")

if __name__ == "__main__":
    main() 