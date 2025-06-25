# 多智能体强化学习评估结果可视化工具

## 功能概述

本工具用于分析和可视化多智能体强化学习在HOK环境中的训练结果，支持VDN、QMIX、QATTEN、QPLEX等算法的性能评估。

## 主要功能

### 1. 单实验分析
- 怪物剩余血量趋势分析
- 造成伤害分布统计
- 胜率和性能指标计算
- 累积胜率变化趋势
- 分段性能分析

### 2. 多实验比较
- 多算法胜率对比
- 平均伤害效率比较
- 性能稳定性评估
- 综合性能雷达图
- 详细比较报告

### 3. 输出功能
- 高质量图表生成 (PNG/PDF/SVG)
- 详细统计报告 (JSON/TXT)
- 实验元数据提取
- 缓存和增量更新

## 使用方法

### 基本用法

```bash
# 分析单个文件
python visualization/visualize_results.py --files results/sacred/hok/monster_last_hp/2025-06-24_monster_lasthp_vdn_env.txt

# 分析最新5个实验结果
python visualization/visualize_results.py --latest 5 --compare

# 按算法筛选并比较
python visualization/visualize_results.py --algorithm vdn --compare --save-report

# 列出可用文件
python visualization/visualize_results.py --list-files
```

### 高级用法

```bash
# 使用配置文件
python visualization/visualize_results.py --config visualization/config_example.json

# 自定义输出目录和格式
python visualization/visualize_results.py --output ./my_results --format pdf

# 显示图表（GUI环境）
python visualization/visualize_results.py --show --latest 3

# 详细模式输出
python visualization/visualize_results.py --verbose --compare --latest 10
```

### 命令行参数详解

#### 文件选择
- `--files/-f`: 指定具体文件路径
- `--dir/-d`: 结果文件目录 (默认: results/sacred/hok/monster_last_hp)
- `--pattern/-p`: 文件匹配模式 (默认: *monster_lasthp*.txt)
- `--latest/-l`: 选择最新N个文件
- `--algorithm/-a`: 按算法筛选 (vdn/qmix/qatten/qplex/all)

#### 分析参数
- `--initial-hp`: 怪物初始血量 (默认: 30000)
- `--compare`: 启用多实验比较模式
- `--single-only`: 仅生成单个实验可视化

#### 输出参数
- `--output/-o`: 输出目录 (默认: ./visualization_results)
- `--show`: 显示图表
- `--save-report`: 保存详细比较报告
- `--format`: 输出格式 (png/pdf/svg)

#### 其他参数
- `--list-files`: 列出可用文件并退出
- `--verbose/-v`: 详细输出
- `--no-cache`: 禁用缓存
- `--config`: 使用配置文件

## 配置文件

可以使用JSON配置文件预设参数，例如：

```json
{
  "dir": "results/sacred/hok/monster_last_hp",
  "latest": 5,
  "compare": true,
  "save_report": true,
  "output": "./my_results",
  "format": "png"
}
```

## 输出文件说明

### 单实验分析输出
- `single_visual_TIMESTAMP.png`: 单实验可视化图表
- `single_visual_TIMESTAMP.txt`: 实验统计摘要

### 比较分析输出
- `comparison_visual_TIMESTAMP.png`: 多实验比较图表
- `comparison_report_TIMESTAMP.json`: 详细比较报告

## 性能指标说明

- **胜率**: 完全击败怪物（血量为0）的比例
- **平均剩余血量**: 怪物在所有评估中的平均剩余血量
- **平均造成伤害**: 平均每次评估造成的伤害
- **伤害效率**: 造成伤害与初始血量的比值
- **稳定性评分**: 基于伤害变异系数的稳定性度量

## 系统要求

- Python 3.7+
- matplotlib >= 3.0
- numpy >= 1.18
- seaborn >= 0.11
- pandas >= 1.0

## 故障排除

### 字体显示问题
如果中文显示异常，请确保系统安装了中文字体，或修改代码中的字体设置。

### 内存不足
处理大文件时如遇内存问题，可使用 `--latest` 参数限制处理文件数量。

### 文件权限问题
确保对输出目录有写权限，脚本会自动创建不存在的目录。

## 示例工作流

1. **快速查看**: `python visualization/visualize_results.py --list-files`
2. **单文件分析**: `python visualization/visualize_results.py --latest 1`
3. **算法比较**: `python visualization/visualize_results.py --algorithm all --compare --latest 5`
4. **生成报告**: `python visualization/visualize_results.py --save-report --compare --latest 10` 