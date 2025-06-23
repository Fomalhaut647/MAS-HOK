# 简介

## 前言

腾讯多智能体迷你环境是一个基于电子游戏构建的开放环境，让研究人员能够仅使用本地算力开发和验证多智能体算法。

在腾讯多智能体迷你环境中，你需要通过算法训练多个英雄与野怪战斗。任务结束时，野怪的剩余血量将作为评估指标。在开发指南中，我们提供了如何将VDN、QMIX、QATTEN和QPLEX四种算法集成到环境中的示例，并展示了一些实验结果。最后，代码包中提供了VDN的示例代码。

## 新闻
[2025/4]Docker镜像安装地址变更。

[2025/2]修复了庄周资源加载错误导致无法正确响应动作的问题。

---

## 环境介绍

### 地图
我们的环境包含智能体英雄和野怪。智能体英雄和野怪的分布如下图所示，蓝点代表智能体英雄，红点代表野怪。任务开始时，智能体英雄和野怪将自动在指定位置生成。
![alt text](./static/img/multi_agent_mini_lv.png)

### 英雄
| 名称 | ID | 血量 | 普攻范围 | 技能类型1 | 技能类型2 | 技能类型3 |
| :--: | --: | :----: | :------------: | :--------: | :--------: | :--------: |
| **庄周** | 11301 | 7738 | 2800 | 方向性技能 | 方向性技能 | 目标技能（自释放） |
| **狄仁杰** | 13301 | 5706 | 8000 | 方向性技能 | 方向性技能 | 方向性技能 |
| **貂蝉** | 14101 | 5609 | 6000 | 方向性技能 | 方向性技能 | 目标技能（自释放） |
| **孙悟空** | 16701 | 7843 | 3000 | 目标技能（自释放） | 方向性技能 | 目标技能（自释放） |
| **曹操** | 12801 | 8185 | 2800 | 方向性技能 | 方向性技能 | 目标技能（自释放） |

### 野怪
<table>
  <tr>
    <th>ID</th>
    <td>12202</td>
  </tr>
  <tr>
    <th>血量</th>
    <td>30000</td>
  </tr>
</table>

[这里](https://kaiwu-assets-1258344700.file.myqcloud.com/fe-assets/kaiwu-doc/open-competition-2024/multiagent/mini-hok-demo.mp4)是一个展示不同怪物的视频。

---

## 环境使用

### 安装要求
> 如果使用Linux系统，可以忽略安装要求，直接进行下一步

1. Windows 10/11
2. Python 3.8或更高版本
3. Docker。如果你的电脑上没有安装Docker，请按照[指南](#docker)完成安装。

### 申请许可证
请填写[腾讯AI竞技场多智能体迷你任务环境许可证申请表](https://docs.qq.com/form/page/DVGR3Vk9Jb29lRW9H)。

收到你的申请信息后，我们将尽快审核。通过审核后，你将通过申请表中提供的邮箱地址收到许可证文件。

### 游戏核心安装
1. 启动Docker并在命令行中输入以下命令：
```shell   
  # 拉取Docker镜像
  docker pull tencentailab/marl-mini:gamecore_20250228
  # 检查镜像ID
  docker images
  # 进入开发容器，将IMAGEID替换为镜像的ID
  docker run -it --rm --name "Env_Name" IMAGEID /bin/bash
  # 查询GameCore容器IP地址
  ifconfig
```
2. 请将`license.dat`文件（[许可证申请步骤中获得的文件](#apply-for-license)）复制到成功启动的Docker容器的`/sgame/`路径中。

### 示例代码安装
1. 启动Docker并在命令行中输入以下命令：
```shell   
  # 拉取Docker镜像
  docker pull tencentailab/marl-mini:20240607
  # 进入开发容器，将IMAGEID替换为镜像的ID
  docker run -it --name "Demo_Name" IMAGEID /bin/bash
  # 克隆github代码
  git clone https://github.com/tencent-ailab/marl-hok.git
```
2. 代码放置目录：`/home/ubuntu/marl-hok`

### 环境启动

#### 游戏核心通信配置
启动环境前，请在示例代码配置文件中配置IP。

配置文件目录：`./src/envs/hok/hok_game/conf/gamecore_conf.json`

查询GameCore容器IP地址，并将示例代码容器配置文件中的endpoint字段修改为**IP地址**:3030。

```shell 
# 打开配置文件
vim ./src/envs/hok/hok_game/conf/gamecore_conf.json
```
```shell
# 配置文件内容
{
    "battlesrv_port": 5555,
    "endpoint": "127.0.0.2:3030",
    "ugc_project_id": 400,
    "level_name": "PVE_1_1",
    "retry_times": 10,
    "retry_sleep_seconds": 1
}
```

#### 开始训练
1. 启动GameCore环境。进入Docker容器后，将在sgame目录下自动执行以下命令：
```shell
./ugc_game_core_server
# 成功启动后，输出将显示：UGC GameCore Server started. listen port: 3030
```
2. 启动示例代码
```shell
cd /home/ubuntu/marl-hok
python3 src/main.py --config="vdn" --env-config="hok" with "env_args.map_name=hok"
# 其中--config参数后跟相应的算法，目前支持VDN算法
```

#### 模型保存
1. 训练期间的模型将保存到路径./results/models

#### 评估
1. 在文件./src/config/default.yaml中设置checkpoint_path:的值为要加载的模型所在的路径

2. 执行
```shell
python3 src/main.py --config="vdn" --env-config="hok" with "env_args.map_name=hok"
# 其中--config参数后跟相应的算法，目前支持VDN算法
```

# 工具安装

## Docker

下面我们将介绍如何在Windows系统上安装和使用Docker。有关Docker的更多信息，请参考[Docker官方文档](https://docs.docker.com/)。

**1. 下载安装包**

官方下载链接：https://www.docker.com/get-started/

**2. 安装**

2.1 打开下载的安装包，使用默认选项进行安装。

![alt text](./static/img/docker_install1.png)

2.2 安装完成后，在桌面上打开Docker Desktop客户端。第一次运行时，需要点击[Accept]同意协议，然后点击[Skip]跳过Docker调查，之后就可以开始运行。

![alt text](./static/img/docker_install2.png)
![alt text](./static/img/docker_install3.png)

2.3 打开Docker并等待一段时间，可以在左下角看到Docker状态为运行中，表示Docker已成功启动。

![alt text](./static/img/docker_running.png) alt="docker_running" width="50%"

**3. 更新WSL 2内核**

如果在第一次运行Docker后看到以下提示，需要更新WSL 2内核。请按照以下步骤操作

![alt text](./static/img/docker_install4.png) 

3.1 访问弹窗中提示的网站（中文页面，可以[点击这里查看](https://docs.microsoft.com/zh-cn/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)），在打开的页面中找到第4步，下载如下所示的安装包。
  ![alt text](./static/img/wsl-1.png)

3.2 下载完成后，运行WSL安装包。
  ![alt text](./static/img/wsl-2.png)
  ![alt text](./static/img/wsl-3.png)

3.3 安装完成后，点击Finish。
  ![alt text](./static/img/wsl-4.png)

3.4 打开Windows系统终端。你可以按`Windows键 + R`组合键打开运行窗口，在运行窗口中输入`cmd`并按回车，Windows系统终端将打开。（或者，你可以在电脑左下角的搜索框中搜索"命令提示符"，然后点击搜索结果进入终端。）

![alt text](./static/img/wsl-6.png)

3.5 将WSL 2设置为默认版本。复制下面的命令，然后将复制的代码粘贴到终端中并按回车。此时，你将在终端中看到操作成功的消息。

```powershell
wsl --set-default-version 2
```

![alt text](./static/img/wsl-8.png)

3.6 最后，执行WSL更新。同样，在终端中输入下面的命令并按回车完成操作。（**注意：此操作必须在Windows 11系统上执行**）

```powershell
wsl --update
```

有关WSL 2的更多信息，请参考[微软官方文档](https://docs.microsoft.com/zh-cn/windows/wsl/install-manual)。

---

## ABS播放器
使用模型完成评估任务后，将生成ABS录制文件。ABS播放文件可以使用腾讯开悟提供的ABS播放器进行查看和可视化分析。

[ABS播放器下载地址](https://drive.weixin.qq.com/s?k=AJEAIQdfAAomyhtflp)

使用说明：
1. 当前ABS播放器仅支持Windows系统，建议在Windows 10上运行。
2. 下载ABS播放器后需要解压，解压路径不能包含中文字符。解压后，双击`ABSTool.exe`文件进行更新，然后即可使用。
3. 获取ABS录制文件后，需要将ABS文件移动到`ABSTool/Replays`目录。如果没有Replays文件夹，请先启动一次`ABSTool.exe`。

> 注意，由于播放器对机器依赖库的要求，如果下载和加载后出现黑屏或蓝屏，可以尝试安装运行时库来修复。运行时库路径：[运行时库下载地址](https://drive.weixin.qq.com/s?k=AJEAIQdfAAoND6j4mw)

![alt text](./static/img/abs_file.png)

![alt text](./static/img/abs_scene.png)

# 算法

## 算法接入模拟
- 算法库参考Pymarl2，源代码：https://github.com/hijkzzz/pymarl2

- 算法库中包含的常见算法
  - 基于价值的方法：
    - [QMIX: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
    - [VDN: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
    - [IQL: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
    - [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
    - [Qatten: Qatten: A general framework for cooperative multiagent reinforcement learning](https://arxiv.org/abs/2002.03939)
    - [QPLEX: Qplex: Duplex dueling multi-agent q-learning](https://arxiv.org/abs/2008.01062)
    - [WQMIX: Weighted QMIX: Expanding Monotonic Value Function Factorisation](https://arxiv.org/abs/2006.10800)
  - Actor Critic方法：
    - [COMA: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
    - [VMIX: Value-Decomposition Multi-Agent Actor-Critics](https://arxiv.org/abs/2007.12306)
    - [LICA: Learning Implicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2007.02529)
    - [DOP: Off-Policy Multi-Agent Decomposed Policy Gradients](https://arxiv.org/abs/2007.12322)
    - [RIIT: Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning.](https://arxiv.org/abs/2102.03479)

- 算法如何与仿真环境交互
  - 在`./src/run/run.py`中，算法通过调用`runner.run`与仿真环境交互
  - 具体的交互部分在`./src/runners/episode_runner.py`中，采样获得的数据存储在`ReplayBuffer`中
  - 采样过程如下：
    - 通过`self.reset()`调用环境接口文件中的`reset()函数`，向仿真发送重置仿真引擎的命令，并初始化参数
    - 在每一帧交互中，通过调用环境接口文件中的`get_state()函数`获得全局状态，通过调用`get_avail_actions()函数`获得智能体的可执行动作，通过调用`get_obs()函数`获得智能体各自的观察
    - 通过`self.mac.select_actions`获得每个智能体的决策动作
    - 通过`self.env.step`调用环境接口文件中的`step()函数`，将智能体的决策动作发送到仿真环境。使用`act_2_cmd()函数`将智能体的动作转换为仿真可执行的命令，从而控制引擎中智能体的相应动作
    - `step()函数`向算法返回当前帧的奖励`reward`和训练终止标志`terminated`
  - 智能体网络更新：
    - 在`./src/run/run.py`中，使用`buffer.sample(args.batch_size)`从`ReplayBuffer`中采样`batch_size轮`训练数据
    - `./src/learners`是更新网络参数的模块。通过在`./src/run/run.py`中调用`learner.train()`将数据发送到网络进行损失计算和参数更新

## 示例算法实验结果
在示例代码中，我们尝试接入VDN、QMIX、QATTEN和QPLEX这四种协作多智能体强化学习算法。实验结果如下：
![alt text](./static/img/Episode.png)
从上图可以看出，随着训练的进行，龙的剩余血量越来越少，不同算法的最终表现并不一致，体现了该环境对不同算法的可比性。

我们可以在服务器`/sgame`路径下获取abs文件，并通过[ABS播放器](#abs-player)进行可视化分析：
![alt text](./static/img/abs_file.png)
![alt text](./static/img/abs_scene.png)

我们发现传统的协作多智能体强化学习可能导致辅助庄周不努力攻击暴君，这可能是由于协作多智能体算法中一直存在的懒惰智能体现象：由于所有智能体共享团队奖励，辅助庄周的作用难以体现，导致混水摸鱼的现象。

这也表明算法在迷你王者环境中仍有改进空间，未来的研究人员可以设计更好的算法来改善这种现象。