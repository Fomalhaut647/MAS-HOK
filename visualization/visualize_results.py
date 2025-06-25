#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Reinforcement Learning Evaluation Results Visualization Tool
Supports analysis and comparison of multiple training results, suitable for VDN algorithm performance evaluation in HOK environment
"""

import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os
import glob
import json
from datetime import datetime
import pandas as pd

def discover_result_files(base_dir):
    """Automatically discover result files"""
    base_path = Path(base_dir)
    pattern = "**/*monster_lasthp*.txt"
    files = list(base_path.glob(pattern))
    
    # Sort by modification time
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return files

def load_hp_data(file_path):
    """Load monster remaining HP data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            hp_data = ast.literal_eval(content)
            
            # Data validation
            if not isinstance(hp_data, (list, tuple)):
                raise ValueError("Data format error: should be list or tuple")
            
            hp_array = np.array(hp_data, dtype=float)
            
            # Check data validity
            if len(hp_array) == 0:
                raise ValueError("Data is empty")
            
            if np.any(hp_array < 0):
                print(f"⚠️ Warning: Found negative HP values, will be corrected to 0")
                hp_array = np.maximum(hp_array, 0)
            
            if np.any(hp_array > 50000):
                print(f"⚠️ Warning: Found abnormally high HP values (>50000)")
            
            return hp_array
            
    except Exception as e:
        print(f"❌ Failed to load file {file_path}: {e}")
        return None

def extract_metadata_from_filename(file_path):
    """Extract metadata from filename"""
    filename = Path(file_path).name
    parts = filename.split('_')
    
    metadata = {
        'filename': filename,
        'filepath': str(file_path),
        'algorithm': 'unknown',
        'environment': 'unknown',
        'timestamp': 'unknown'
    }
    
    # Try to parse timestamp
    if len(parts) >= 2:
        try:
            timestamp_str = f"{parts[0]} {parts[1]}"
            metadata['timestamp'] = timestamp_str
        except:
            pass
    
    # Try to parse algorithm and environment
    for part in parts:
        if 'vdn' in part.lower():
            metadata['algorithm'] = 'VDN'
        elif 'qmix' in part.lower():
            metadata['algorithm'] = 'QMIX'
        elif 'qatten' in part.lower():
            metadata['algorithm'] = 'QATTEN'
        elif 'qplex' in part.lower():
            metadata['algorithm'] = 'QPLEX'
        
        if 'hok' in part.lower():
            metadata['environment'] = 'HOK'
        elif 'env' in part.lower():
            metadata['environment'] = 'Environment'
    
    return metadata

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
        'damage_std': np.std(damage_dealt),
        'median_remaining_hp': np.median(hp_data),
        'damage_efficiency': np.mean(damage_dealt) / initial_hp,
        'stability_score': 1 - (np.std(damage_dealt) / np.mean(damage_dealt)) if np.mean(damage_dealt) > 0 else 0
    }
    
    return stats, damage_dealt, damage_ratio

def create_single_visualizations(hp_data, damage_dealt, damage_ratio, stats, metadata, save_dir='./visualization_results'):
    """Create visualization charts for single experiment"""
    Path(save_dir).mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. HP time series
    plt.subplot(3, 3, 1)
    plt.plot(hp_data, alpha=0.8, linewidth=1, color='red')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Complete Defeat Line')
    plt.title(f'Monster Remaining HP Trend - {metadata["algorithm"]}', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Remaining HP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. HP distribution
    plt.subplot(3, 3, 2)
    plt.hist(hp_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=stats['avg_remaining_hp'], color='red', linestyle='--', 
                label=f'Average: {stats["avg_remaining_hp"]:.0f}')
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
                label=f'Average: {stats["avg_damage_ratio"]:.1f}%')
    plt.title('Damage Ratio Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Damage Ratio (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Moving average performance
    window_size = min(100, len(hp_data) // 5)
    if len(hp_data) >= window_size and window_size > 1:
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
        if len(segment_data) > 0:
            segment_win_rate = np.sum(segment_data == 0) / len(segment_data) * 100
            segment_stats.append(segment_win_rate)
        else:
            segment_stats.append(0)
    
    plt.bar(range(1, len(segment_stats) + 1), segment_stats, alpha=0.7, color='orange')
    plt.title('Segmented Win Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Training Phase')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # 8. Box plot
    plt.subplot(3, 3, 8)
    box_data = [hp_data, damage_dealt]
    box_plot = plt.boxplot(box_data)
    plt.xticks([1, 2], ['Remaining HP', 'Damage Dealt'])
    plt.title('Data Distribution Box Plot', fontsize=14, fontweight='bold')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # 9. Performance summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    Performance Summary Report - {metadata['algorithm']}
    Timestamp: {metadata['timestamp']}
    
    Total Evaluations: {stats['total_evaluations']:,}
    Complete Defeats: {stats['complete_defeats']:,}
    Win Rate: {stats['win_rate']:.2f}%
    
    Average Remaining HP: {stats['avg_remaining_hp']:.0f}
    Average Damage Dealt: {stats['avg_damage']:.0f}
    Average Damage Ratio: {stats['avg_damage_ratio']:.2f}%
    
    Best Performance: {stats['min_remaining_hp']:.0f} Remaining HP
    Worst Performance: {stats['max_remaining_hp']:.0f} Remaining HP
    Performance Stability: {stats['damage_std']:.0f} (Damage Std)
    Damage Efficiency: {stats['damage_efficiency']:.3f}
    Stability Score: {stats['stability_score']:.3f}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    
    # Save file
    safe_filename = f"eval_visual_{metadata['algorithm']}_{metadata['timestamp'].replace(' ', '_').replace(':', '-')}.png"
    save_path = Path(save_dir) / safe_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Single experiment visualization saved to: {save_path}")
    
    # Generate brief summary text file
    summary_path = save_path.with_suffix('.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Analysis Summary - {metadata['algorithm']}\n")
        f.write("=" * 50 + "\n")
        f.write(f"File: {metadata['filename']}\n")
        f.write(f"Algorithm: {metadata['algorithm']}\n")
        f.write(f"Timestamp: {metadata['timestamp']}\n")
        f.write(f"Total Evaluations: {stats['total_evaluations']}\n")
        f.write(f"Win Rate: {stats['win_rate']:.2f}%\n")
        f.write(f"Average Remaining HP: {stats['avg_remaining_hp']:.0f}\n")
        f.write(f"Average Damage Dealt: {stats['avg_damage']:.0f}\n")
        f.write(f"Damage Efficiency: {stats['damage_efficiency']:.3f}\n")
        f.write(f"Stability Score: {stats['stability_score']:.3f}\n")
    
    return fig

def create_comparison_visualizations(experiments_data, save_dir='./visualization_results'):
    """Create multi-experiment comparison visualization"""
    Path(save_dir).mkdir(exist_ok=True)
    
    if len(experiments_data) < 2:
        print("⚠️ At least 2 experiment data required for comparison")
        return None
    
    sns.set_style("whitegrid")
    plt.style.use('default')
    
    fig = plt.figure(figsize=(20, 16))
    
    # Prepare comparison data
    comparison_stats = []
    # Use more compatible color scheme
    import matplotlib.colors as mcolors
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_list[i % len(color_list)] for i in range(len(experiments_data))]
    
    for exp_data in experiments_data:
        comparison_stats.append({
            'label': f"{exp_data['metadata']['algorithm']} ({exp_data['metadata']['timestamp']})",
            'stats': exp_data['stats'],
            'hp_data': exp_data['hp_data'],
            'damage_dealt': exp_data['damage_dealt'],
            'damage_ratio': exp_data['damage_ratio'],
            'metadata': exp_data['metadata']
        })
    
    # 1. Win rate comparison
    plt.subplot(3, 3, 1)
    win_rates = [exp['stats']['win_rate'] for exp in comparison_stats]
    labels = [exp['label'] for exp in comparison_stats]
    bars = plt.bar(range(len(win_rates)), win_rates, color=colors)
    plt.title('Win Rate Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate (%)')
    plt.xticks(range(len(labels)), [f"Exp{i+1}" for i in range(len(labels))], rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.1f}%', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 2. Average damage comparison
    plt.subplot(3, 3, 2)
    avg_damages = [exp['stats']['avg_damage'] for exp in comparison_stats]
    bars = plt.bar(range(len(avg_damages)), avg_damages, color=colors)
    plt.title('Average Damage Dealt Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Average Damage')
    plt.xticks(range(len(labels)), [f"Exp{i+1}" for i in range(len(labels))], rotation=45)
    
    for bar, damage in zip(bars, avg_damages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{damage:.0f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 3. HP time series comparison
    plt.subplot(3, 3, 3)
    for i, exp in enumerate(comparison_stats):
        hp_data = exp['hp_data']
        # Sample or truncate sequences of different lengths
        if len(hp_data) > 1000:
            indices = np.linspace(0, len(hp_data)-1, 1000, dtype=int)
            hp_sampled = hp_data[indices]
            x_sampled = indices
        else:
            hp_sampled = hp_data
            x_sampled = range(len(hp_data))
        
        plt.plot(x_sampled, hp_sampled, alpha=0.7, color=colors[i], 
                label=f'Exp{i+1}', linewidth=1)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Complete Defeat Line')
    plt.title('HP Trend Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Remaining HP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Performance stability comparison
    plt.subplot(3, 3, 4)
    stability_scores = [exp['stats']['stability_score'] for exp in comparison_stats]
    bars = plt.bar(range(len(stability_scores)), stability_scores, color=colors)
    plt.title('Performance Stability Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Stability Score')
    plt.xticks(range(len(labels)), [f"Exp{i+1}" for i in range(len(labels))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Damage efficiency comparison
    plt.subplot(3, 3, 5)
    damage_efficiencies = [exp['stats']['damage_efficiency'] for exp in comparison_stats]
    bars = plt.bar(range(len(damage_efficiencies)), damage_efficiencies, color=colors)
    plt.title('Damage Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Damage Efficiency')
    plt.xticks(range(len(labels)), [f"Exp{i+1}" for i in range(len(labels))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Cumulative win rate comparison
    plt.subplot(3, 3, 6)
    for i, exp in enumerate(comparison_stats):
        hp_data = exp['hp_data']
        wins = (hp_data == 0).astype(int)
        cumulative_win_rate = np.cumsum(wins) / np.arange(1, len(wins) + 1) * 100
        
        # Sampling handling
        if len(cumulative_win_rate) > 1000:
            indices = np.linspace(0, len(cumulative_win_rate)-1, 1000, dtype=int)
            cumulative_sampled = cumulative_win_rate[indices]
            x_sampled = indices
        else:
            cumulative_sampled = cumulative_win_rate
            x_sampled = range(len(cumulative_win_rate))
        
        plt.plot(x_sampled, cumulative_sampled, color=colors[i], 
                label=f'Exp{i+1}', linewidth=2)
    
    plt.title('Cumulative Win Rate Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Cumulative Win Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Damage distribution comparison (box plot)
    plt.subplot(3, 3, 7)
    damage_data = [exp['damage_dealt'] for exp in comparison_stats]
    exp_labels = [f"Exp{i+1}" for i in range(len(comparison_stats))]
    
    box_plot = plt.boxplot(damage_data, patch_artist=True)
    plt.xticks(range(1, len(exp_labels) + 1), exp_labels)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Damage Distribution Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Damage Dealt')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 8. Comprehensive performance radar chart
    plt.subplot(3, 3, 8, projection='polar')
    
    # Define evaluation dimensions
    categories = ['Win Rate', 'Damage Efficiency', 'Stability', 'Consistency']
    N = len(categories)
    
    # Calculate angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the shape
    
    for i, exp in enumerate(comparison_stats):
        # Normalize data to 0-1 range
        values = [
            exp['stats']['win_rate'] / 100,  # Win rate
            exp['stats']['damage_efficiency'],  # Damage efficiency
            exp['stats']['stability_score'],  # Stability
            1 - (exp['stats']['damage_std'] / 30000)  # Consistency (reverse std)
        ]
        values += values[:1]  # Close the shape
        
        plt.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                label=f'Exp{i+1}', alpha=0.7)
        plt.fill(angles, values, alpha=0.25, color=colors[i])
    
    plt.xticks(angles[:-1], categories)
    plt.ylim(0, 1)
    plt.title('Comprehensive Performance Radar Chart', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 9. Detailed comparison table
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create comparison table data
    table_data = []
    for i, exp in enumerate(comparison_stats):
        stats = exp['stats']
        table_data.append([
            f"Exp{i+1}",
            f"{stats['win_rate']:.1f}%",
            f"{stats['avg_damage']:.0f}",
            f"{stats['damage_efficiency']:.3f}",
            f"{stats['stability_score']:.3f}"
        ])
    
    headers = ['Experiment', 'Win Rate', 'Avg Damage', 'Damage Efficiency', 'Stability']
    
    # Create table
    table = plt.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set table style
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Comparison Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_dir) / f"comparison_visual_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison visualization saved to: {save_path}")
    
    return fig

def print_detailed_stats(stats, metadata):
    """Print detailed statistics"""
    print("\n" + "="*70)
    print(f"    {metadata['algorithm']} Algorithm HOK Environment Evaluation Report")
    print(f"    Timestamp: {metadata['timestamp']}")
    print("="*70)
    print(f"📊 Total Evaluations: {stats['total_evaluations']:,}")
    print(f"🏆 Complete Defeats: {stats['complete_defeats']:,}")
    print(f"📈 Win Rate: {stats['win_rate']:.2f}%")
    print(f"❤️  Average Remaining HP: {stats['avg_remaining_hp']:.0f}")
    print(f"💔 Median Remaining HP: {stats['median_remaining_hp']:.0f}")
    print(f"⚔️  Average Damage Dealt: {stats['avg_damage']:.0f}")
    print(f"💪 Average Damage Ratio: {stats['avg_damage_ratio']:.2f}%")
    print(f"🎯 Best Performance: {stats['min_remaining_hp']:.0f} Remaining HP")
    print(f"😅 Worst Performance: {stats['max_remaining_hp']:.0f} Remaining HP")
    print(f"📊 Performance Stability: {stats['damage_std']:.0f} (Damage Std)")
    print(f"⚡ Damage Efficiency: {stats['damage_efficiency']:.3f}")
    print(f"🔄 Stability Score: {stats['stability_score']:.3f}")
    print("="*70)

def save_comparison_report(experiments_data, save_dir):
    """Save comparison report to JSON file"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }
    
    for exp_data in experiments_data:
        exp_report = {
            'metadata': exp_data['metadata'],
            'stats': exp_data['stats'],
            'summary': {
                'total_episodes': len(exp_data['hp_data']),
                'data_range': [float(np.min(exp_data['hp_data'])), float(np.max(exp_data['hp_data']))],
                'quartiles': [float(q) for q in np.percentile(exp_data['hp_data'], [25, 50, 75])]
            }
        }
        # Convert numpy types to Python native types
        for key, value in exp_report['stats'].items():
            if isinstance(value, np.floating):
                exp_report['stats'][key] = float(value)
            elif isinstance(value, np.integer):
                exp_report['stats'][key] = int(value)
        
        report['experiments'].append(exp_report)
    
    # Add comparison analysis
    if len(experiments_data) > 1:
        win_rates = [exp['stats']['win_rate'] for exp in experiments_data]
        damage_effs = [exp['stats']['damage_efficiency'] for exp in experiments_data]
        stability_scores = [exp['stats']['stability_score'] for exp in experiments_data]
        
        report['comparison_analysis'] = {
            'best_win_rate': {
                'experiment_index': int(np.argmax(win_rates)),
                'value': float(np.max(win_rates))
            },
            'best_damage_efficiency': {
                'experiment_index': int(np.argmax(damage_effs)),
                'value': float(np.max(damage_effs))
            },
            'best_stability': {
                'experiment_index': int(np.argmax(stability_scores)),
                'value': float(np.max(stability_scores))
            },
            'performance_variance': {
                'win_rate_std': float(np.std(win_rates)),
                'damage_efficiency_std': float(np.std(damage_effs)),
                'stability_std': float(np.std(stability_scores))
            }
        }
    
    # Save report
    report_path = Path(save_dir) / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📄 Comparison report saved to: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Reinforcement Learning Evaluation Results Visualization Tool',
                                   formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog="""
Usage Examples:
  python visualize_results.py --latest 2 --compare                    # Compare latest 2 experiments
  python visualize_results.py --files file1.txt file2.txt --compare   # Compare specified files
  python visualize_results.py --algorithm vdn --latest 5              # Analyze latest 5 VDN algorithm experiments
  python visualize_results.py --list-files                           # List available files
  python visualize_results.py --dir results/sacred/hok/monster_last_hp # Specify directory
""")
    
    # File selection parameters
    parser.add_argument('--files', '-f', nargs='*', 
                        help='Specify result file paths to analyze (can specify multiple)')
    parser.add_argument('--dir', '-d', 
                        default='results/sacred/hok/monster_last_hp',
                        help='Result file directory (default: results/sacred/hok/monster_last_hp)')
    parser.add_argument('--pattern', '-p', 
                        default='*monster_lasthp*.txt',
                        help='File matching pattern (default: *monster_lasthp*.txt)')
    parser.add_argument('--latest', '-l', type=int, 
                        help='Select latest N files for analysis')
    parser.add_argument('--algorithm', '-a', 
                        choices=['vdn', 'qmix', 'qatten', 'qplex', 'all'],
                        help='Filter files by algorithm')
    
    # Analysis parameters
    parser.add_argument('--initial-hp', type=int, default=30000,
                        help='Monster initial HP (default: 30000)')
    parser.add_argument('--compare', action='store_true',
                        help='Enable multi-experiment comparison mode')
    parser.add_argument('--single-only', action='store_true',
                        help='Generate only single experiment visualizations')
    
    # Output parameters
    parser.add_argument('--output', '-o', default='./visualization_results',
                        help='Output directory (default: ./visualization_results)')
    parser.add_argument('--show', action='store_true',
                        help='Display charts')
    parser.add_argument('--save-report', action='store_true',
                        help='Save detailed comparison report')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='Output image format (default: png)')
    
    # Other parameters
    parser.add_argument('--list-files', action='store_true',
                        help='List available result files and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Set matplotlib backend
    if not args.show:
        plt.switch_backend('Agg')  # Non-GUI backend, suitable for server environments
    
    try:
        # Discover and select files
        if args.files:
            # User specified files
            selected_files = [Path(f) for f in args.files if Path(f).exists()]
            if len(selected_files) != len(args.files):
                missing = [f for f in args.files if not Path(f).exists()]
                print(f"⚠️ The following files do not exist: {missing}")
        else:
            # Auto discover files
            all_files = discover_result_files(args.dir)
            
            if not all_files:
                print(f"❌ No files matching pattern {args.pattern} found in directory {args.dir}")
                return
            
            # Filter by algorithm
            if args.algorithm and args.algorithm != 'all':
                algo_files = []
                for f in all_files:
                    if args.algorithm.lower() in f.name.lower():
                        algo_files.append(f)
                all_files = algo_files
            
            # Select latest N files
            if args.latest:
                selected_files = all_files[:args.latest]
            else:
                selected_files = all_files
        
        if args.list_files:
            print("\n📋 Available result files:")
            for i, f in enumerate(selected_files, 1):
                metadata = extract_metadata_from_filename(f)
                print(f"{i:2d}. {f.name}")
                print(f"    Algorithm: {metadata['algorithm']}, Time: {metadata['timestamp']}")
                print(f"    Path: {f}")
            return
        
        if not selected_files:
            print("❌ No files found matching criteria")
            return
        
        if args.verbose:
            print(f"📁 Will analyze {len(selected_files)} files:")
            for f in selected_files:
                print(f"  - {f.name}")
        
        # Load and analyze data
        experiments_data = []
        
        for file_path in selected_files:
            print(f"\n🔄 Processing: {file_path.name}")
            
            # Load data
            hp_data = load_hp_data(file_path)
            if hp_data is None:
                continue
            
            # Extract metadata
            metadata = extract_metadata_from_filename(file_path)
            
            # Analyze performance
            stats, damage_dealt, damage_ratio = analyze_performance(hp_data, args.initial_hp)
            
            # Print statistics
            if args.verbose:
                print_detailed_stats(stats, metadata)
            
            # Store experiment data
            experiments_data.append({
                'hp_data': hp_data,
                'damage_dealt': damage_dealt,
                'damage_ratio': damage_ratio,
                'stats': stats,
                'metadata': metadata
            })
            
            # Generate single experiment visualization
            if not args.single_only:
                create_single_visualizations(hp_data, damage_dealt, damage_ratio, 
                                           stats, metadata, args.output)
        
        if not experiments_data:
            print("❌ No experiment data loaded successfully")
            return
        
        print(f"\n✅ Successfully loaded {len(experiments_data)} experiment data")
        
        # Generate comparison visualization
        if args.compare and len(experiments_data) >= 2:
            print("\n🔄 Generating comparison visualization...")
            create_comparison_visualizations(experiments_data, args.output)
            
            if args.save_report:
                save_comparison_report(experiments_data, args.output)
        elif args.compare and len(experiments_data) < 2:
            print("⚠️ At least 2 experiment data required for comparison")
        
        # Display charts
        if args.show:
            plt.show()
        
        print(f"\n🎉 Visualization analysis complete! Results saved in {args.output} directory")
        
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted operation")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 