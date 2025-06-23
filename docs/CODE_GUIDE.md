## 代码目录介绍
```python
├── src
│   ├── components
│   │   ├── action_selectors.py
│   │   ├── episode_buffer.py
│   │   ├── epsilon_schedules.py
│   │   ├── __init__.py
│   │   ├── segment_tree.py
│   │   └── transforms.py
│   ├── config
│   │   ├── algs
│   │   │   └── vdn.yaml
│   │   ├── default.yaml
│   │   └── envs
│   │       └── hok.yaml
│   ├── controllers
│   │   ├── basic_controller.py
│   │   ├── __init__.py
│   │   └── n_controller.py
│   ├── envs
│   │   ├── hok
│   │   │   ├── hok_env.py
│   │   │   ├── hok_game
│   │   │   │   ├── agent
│   │   │   │   │   ├── actor.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── client
│   │   │   │   │   ├── gamecore_controller.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── conf
│   │   │   │   │   ├── config.py
│   │   │   │   │   ├── gamecore_conf.json
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── natureclient_conf.json
│   │   │   │   ├── __init__.py
│   │   │   │   ├── nature_client.py
│   │   │   │   ├── protocol
│   │   │   │   │   ├── command.proto
│   │   │   │   │   ├── common.proto
│   │   │   │   │   ├── easy.proto
│   │   │   │   │   ├── hero.proto
│   │   │   │   │   ├── python
│   │   │   │   │   │   ├── build_py.sh
│   │   │   │   │   │   ├── command_pb2.py
│   │   │   │   │   │   ├── common_pb2.py
│   │   │   │   │   │   ├── hero_pb2.py
│   │   │   │   │   │   ├── scene_pb2.py
│   │   │   │   │   │   ├── sgame_ai_server_pb2.py
│   │   │   │   │   │   └── sgame_state_pb2.py
│   │   │   │   │   ├── scene.proto
│   │   │   │   │   ├── sgame_ai_server.proto
│   │   │   │   │   └── sgame_state.proto
│   │   │   │   └── README.md
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   └── multiagentenv.py
│   ├── __init__.py
│   ├── learners
│   │   ├── __init__.py
│   │   └── nq_learner.py
│   ├── main.py
│   ├── modules
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   └── n_rnn_agent.py
│   │   ├── __init__.py
│   │   └── mixers
│   │       ├── __init__.py
│   │       ├── nmix.py
│   │       ├── qatten.py
│   │       └── vdn.py
│   ├── run
│   │   ├── __init__.py
│   │   └── run.py
│   ├── runners
│   │   ├── episode_runner.py
│   │   └── __init__.py
│   └── utils
│       ├── logging.py
│       ├── rl_utils.py
│       ├── th_utils.py
│       └── timehelper.py
└── train.sh
主要文件说明如下:
- 此环境中的代码借鉴了SMAC环境的代码实现。
- `./src/envs/hok/hok_game/client/gamecore_controller.py` 负责控制引擎，通过向ugc_game_core_server发送HTTP请求来启动/停止gamecore。
- `./src/envs/hok/hok_game/conf/gamecore_conf.json` 用于配置服务器的IP和端口访问。
- `./src/envs/hok/hok_game/conf/gamecore_conf.json` 用于设置英雄属性和configID。


## 环境接口文件

强化学习算法将通过环境接口文件与gamecore进行交互。

### 环境类属性参数

```python
# 继承MultiAgentEnv类和腾讯GameCore类
class HokEnv(MultiAgentEnv, NatureClient):
    def __init__(
        self,
        map_name='hok',
        num_agents=5,
        time_step=0,
        seed=123,
        episode_limit=150,
        client_id=f"debug-train",
        logger=None,
        max_frame=60000
    ):
        super().__init__(client_id ,logger, max_frame)
```

> 注意：以上是初始化的默认参数
>- `NatureClient` 是腾讯提供的GameCore规则类
>- `MultiAgentEnv` 是多智能体强化学习的基本环境类

#### 地图相关参数
```python
self.map_name = map_name # 地图名称
self.n_agents = num_agents # 建模的智能体数量
```

#### 训练相关参数
```python
self.episode_limit = episode_limit # episode_limit的最大步数
self._episode_steps = 0 # 每个episode的步数计数
```

#### 动作相关参数
```python
self.action_space = 13 # 动作空间维度
self.n_actions = self.action_space
```

> 注意：
>- 动作包括移动动作：上、下、左、右、左上、右上、右下、左下
>- 技能动作：普攻、1、2、3、召唤师技能（使用）

#### 状态空间参数
```python
self.observation_space = 6 # 每个智能体的观测空间为6

# 初始化状态和观测变量
self.obs = None
self.state = None
```

> 注意：
>- 每个智能体的观测为：[自身x坐标, 自身z坐标, 自身血量, 暴君x坐标, 暴君z坐标, 暴君血量]

#### 其他相关参数
```python
self.seed = seed # 随机种子
self.monster_hp_total = [] # 存储当前episode暴君的血量
self.pos_delta = 1500 # 离散化移动动作后的偏移量，基于原坐标移动1500
self.SKILL_TYPE = ["obj_skill", "dir_skill", "pos_skill", "talent_skill"] # 动作执行方式
```

> 注意：
>- self.SKILL_TYPE是技能的动作执行方式
>- obj_skill是指向目标的执行方式，当自己释放时，target应该是英雄自己的ID
>- dir_skill是指向方向的执行方式，默认朝向暴君方向释放
>- pos_skill是指向位置的执行方式，目前此设置不涉及此技能类型
>- talent_skill是召唤师技能，target应该是英雄自己的ID

### 获取状态函数
```python
def get_obs(self):
    """在二维列表中返回所有智能体的观测。"""
    agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
    return agents_obs
    
def get_obs_agent(self, agent_id):
    return self.obs[agent_id].reshape(-1)

def get_obs_size(self):
    """返回观测的大小"""
    return self.observation_space

def get_global_state(self):
    """返回全局状态"""
    return self.obs.flatten()
def get_state(self):
    """返回全局状态。"""
    return self.get_global_state()

def get_state_size(self):
   """返回全局状态的大小。"""
   return self.get_obs_size() * self.n_agents
```
### 获取可用动作
```python
def get_avail_actions(self):
    """在列表中返回所有智能体的可用动作。"""
    ## 现在已经添加了技能，不是所有技能都能使用，5678技能键需要检查calm_down==0，如果为0，则为1
    all_actor_legal_skill = []
    for hero in self.heroes:
        if hero.actor_id == 6: # 暴君直接退出
            break 
        hero_legal_skill = [1,1,1,1] # 方向移动不需要考虑calm_down
        # hero是当前英雄，只选择合法技能
        cur_legal_skill = [1 if slot.cooldown==0 else 0 for slot in hero.skill_state.slot_states]
        hero_legal_skill.extend(cur_legal_skill)
        hero_legal_skill.extend([1,1,1,1]) # 方向移动不需要考虑calm_down
        all_actor_legal_skill.append(hero_legal_skill)
    
    return all_actor_legal_skill
        
def get_avail_agent_actions(self, agent_id):
    """返回agent_id的可用动作。"""
    return self.get_avail_actions()[agent_id]

def get_total_actions(self):
    """返回智能体可能执行的动作总数"""
    return self.action_space
```
> 注意：
>- 过滤智能体在当前帧可以执行的合法动作
>- 为每个智能体提供一个长度等于动作空间的列表，其中每个索引表示是否可以执行该动作，0/1分别表示不能/可以执行

### 动作转换函数
#### 技能定义相关
```python
def get_skill_id_by_slot_type(self, hero, slot_type):
    """通过slot_type获取技能槽
    
    params:
        hero: 英雄
        slot_type: 技能1 / 2 / 3
        
    return:    
        技能ID
    """
    skill_id = [slot.configId for slot in hero.skill_state.slot_states if slot.slot_type == slot_type][0]
    return skill_id
        
def __obj_skill_command(self, target, hero, skill_slot_type):
    """指向目标技能
    
    params:
        target: 目标ID
        hero: 英雄ID
        skill_slot_type: 技能
    
    return:
        指向目标技能命令
    
    """
    cmd_pkg = CmdPkg()
    objSkill = ObjSkill()
    objSkill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
    objSkill.actorID = target
    objSkill.slotType = skill_slot_type
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_ObjSkill
    cmd_pkg.obj_skill.CopyFrom(objSkill)

    return cmd_pkg
        
def __dir_skill_command(self, target, hero, skill_slot_type):
    """指向方向技能
    
    params:
        target: 目标ID
        hero: 英雄ID
        skill_slot_type: 技能
    
    return:
        指向方向技能命令
    
    """
    cmd_pkg = CmdPkg()
    skill = DirSkill() 
    skill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
    skill.actorID = target 
    skill.slotType = skill_slot_type
    skill.degree = 4
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_DirSkill
    cmd_pkg.dir_skill.CopyFrom(skill)
    
    return cmd_pkg
        
def __pos_skill_command(self, hero, skill_slot_type):
    """指向位置技能
    
    params:
        hero: 英雄ID
        skill_slot_type: 技能
    
    return:
        指向位置技能命令
    
    """
    cmd_pkg = CmdPkg()
    skill = PosSkill()
    skill.skillID = self.get_skill_id_by_slot_type(hero, skill_slot_type)
    skill.destPos.CopyFrom(self.random_position())
    skill.slotType = skill_slot_type
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_PosSkill
    cmd_pkg.pos_skill.CopyFrom(skill)
    
    return cmd_pkg
        
def __talent_skill_command(self, target):
    """召唤师技能
    
    params:
        hero: 英雄ID
        skill_slot_type: 技能
    
    return:
        召唤师技能命令
    
    """
    cmd_pkg = CmdPkg()
    skill = TalentSkill()
    skill.degree = 90
    skill.actorID = target
    cmd_pkg.command_type = CommandType.COMMAND_TYPE_TalentSkill
    cmd_pkg.talent_skill.CopyFrom(skill)
    
    return cmd_pkg
```
#### 动作转命令
```python
def act_2_cmd(self, actions):
    """将智能体动作转换为gamecore可以执行的命令
    
    params:
        actions: 智能体动作列表 -> [2,3,1,4,1]
    
    return:
        为每个智能体生成命令列表
    """
    cmd_list, stop_game = [], False
    for _id, act in enumerate(actions): # [0,1,2,3,4]
        '''actor_id = _id+1'''
        cmd_pkg = CmdPkg()
        # 适配每个英雄的技能释放类型
        if _id == 0:
            """庄周"""
            if act == 4:
                target = 6
            elif act == 5:
                skill_type = 'dir_skill'
                target = 6
            elif act == 6:
                skill_type = 'obj_skill'
                target = _id+1  
            elif act == 7:
                skill_type = 'obj_skill'
                target = _id+1  
            elif act == 8:
                skill_type = 'talent_skill'
                target = _id+1  
        elif _id == 1:
            """狄仁杰"""
            if act == 4:
                target = 6
            elif act == 5:
                skill_type = 'dir_skill'
                target = 6
            elif act == 6:
                skill_type = 'dir_skill'
                target = 6
            elif act == 7:
                skill_type = 'dir_skill'
                target = 6
            elif act == 8:
                skill_type = 'talent_skill'
                target = _id+1
        elif _id == 2:
            """貂蝉"""
            if act == 4:
                target = 6
            elif act == 5:
                skill_type = 'dir_skill'
                target = 6
            elif act == 6:
                skill_type = 'dir_skill'
                target = 6
            elif act == 7:
                skill_type = 'obj_skill'
                target = _id+1
            elif act == 8:
                skill_type = 'talent_skill'
                target = _id+1
        elif _id == 3:
            """孙悟空"""
            if act == 4:
                target = 6
            elif act == 5:
                skill_type = 'obj_skill'
                target = _id+1
            elif act == 6:
                skill_type = 'dir_skill'
                target = 6
            elif act == 7:
                skill_type = 'obj_skill'
                target = _id+1
            elif act == 8:
                skill_type = 'talent_skill'
                target = _id+1
        elif _id == 4:
            """曹操"""
            if act == 4:
                target = 6
            elif act == 5:
                skill_type = 'dir_skill'
                target = 6
            elif act == 6:
                skill_type = 'dir_skill'
                target = 6
            elif act == 7:
                skill_type = 'obj_skill'
                target = _id+1
            elif act == 8:
                skill_type = 'talent_skill'
                target = _id+1
        
        hero = self.heroes[_id]
        if act == 4:
            # 普通攻击
            attack = AttackCommon()
            attack.actorID = target
            attack.start = 1
            cmd_pkg.command_type = CommandType.COMMAND_TYPE_AttackCommon
            cmd_pkg.attack_common.CopyFrom(attack)

        elif act == 5:
            skill_slot_type = 1 # 技能1
            if skill_type == "obj_skill":
                cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                cmd_pkg = self.__talent_skill_command(target)

        elif act == 6:
            skill_slot_type = 2 
            if skill_type == "obj_skill":
                cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                cmd_pkg = self.__talent_skill_command(target)

        elif act == 7:
            skill_slot_type = 3 
            if skill_type == "obj_skill":
                cmd_pkg = self.__obj_skill_command(target, hero, skill_slot_type)
            elif skill_type == "dir_skill":
                cmd_pkg = self.__dir_skill_command(target, hero, skill_slot_type)
            elif skill_type == "pos_skill":
                cmd_pkg = self.__pos_skill_command(hero, skill_slot_type)
            elif skill_type == "talent_skill":
                cmd_pkg = self.__talent_skill_command(target)
        
        elif act == 8: 
            '''obj_skill talent skill __talent_skill_command(target)'''
            cmd_pkg = self.__talent_skill_command(target)
        
        else:
            # # 0 上 1 下 2 左 3 右 9 左上 10 右上 11 右下 12 左下
            move_pos = MoveToPos()
            move_pos.destPos.CopyFrom(self.position_change(_id, act))
            cmd_pkg.command_type = CommandType.COMMAND_TYPE_MovePos
            cmd_pkg.move_pos.CopyFrom(move_pos)

        cmd = AICommandInfo(
            actor_id=int(_id+1),
            cmd_info=cmd_pkg
        )
        cmd_list.append(cmd)

    return cmd_list, stop_game
```

1. 动作映射关系：
<table>
  <tr>
    <th>0</th>
    <td>上</td>
  </tr>
  <tr>
    <th>1</th>
    <td>下</td>
  </tr>
  <tr>
    <th>2</th>
    <td>左</td>
  </tr>
  <tr>
    <th>3</th>
    <td>右</td>
  </tr>
  <tr>
    <th>4</th>
    <td>普通攻击</td>
  </tr>
  <tr>
    <th>5</th>
    <td>技能1 目前所有1、2、3技能都以obj_skill形式释放</td>
  </tr>
  <tr>
    <th>6</th>
    <td>技能2</td>
  </tr>
  <tr>
    <th>7</th>
    <td>技能3</td>
  </tr>
  <tr>
    <th>8</th>
    <td>技能4</td>
  </tr>
  <tr>
    <th>9</th>
    <td>左上</td>
  </tr>
  <tr>
    <th>10</th>
    <td>右上</td>
  </tr>
  <tr>
    <th>11</th>
    <td>右下</td>
  </tr>
  <tr>
    <th>12</th>
    <td>左下</td>
  </tr>
</table>
         
2. actions = [a1, a2, a3, a4, a5] 对应5个智能体：
   - 庄周：技能1（方向性），技能2（自身释放），技能3（自身释放）        
   - 狄仁杰：技能1（方向性），技能2（方向性），技能3（方向性）               
   - 貂蝉：技能1（方向性），技能2（方向性），技能3（自身释放）               
   - 孙悟空：技能1（自身释放），技能2（方向性），技能3（自身释放）                   
   - 曹操：技能1（方向性），技能2（方向性），技能3（自身释放）

### 环境初始化函数 reset()
```python
def reset(self, if_test=False, args=None, cur_time=None):
    """重置环境。在每个完整episode后必须调用。
        返回初始观测和状态。
        在采样的每个episode阶段开始游戏
    """
    # 确定游戏的胜利条件
    self.battles_won = 0
    self.battles_game = 0
    # 存储每帧暴君的血量
    self.monster_hp_total = [] 
    # 游戏的初始步数为0
    self._episode_steps = 0
    # 每帧返回的附加信息
    self.info_re = {'win_rate': 0., 'test_win_rate': 0., 'battle_won': None, 'monster_last_hp': None}
    
    ## 启动游戏引擎
    # 记录开始时间节点
    self.start_point = time.time()
    # 定义本次游戏启动的id，用于生成abs文件
    if if_test:
        self.game_id = f"{args.name}-hok-test-{cur_time}" 
    else:
        self.game_id = f"{args.name}-hok-train-{cur_time}"
    
    # logger日志记录
    self.logger.info('新episode游戏开始！')
    # 初始化游戏
    self.reset_game()
    t = time.time()
    ret = self.controller.start_game()
    if not ret:
        self.logger.error("启动游戏失败")
        try:
            ret = self.controller.start_game()
            if not ret:
                raise Exception("重试后启动游戏仍然失败")
        except Exception as e:
            self.logger.error(str(e))

    self.logger.debug(f"Nature client 启动游戏耗时 = {time.time() - t} s")
    ti = time.time() - self.start_point
    # 记录游戏启动时间
    self.step_time_cost+=ti
    # 接收初始帧游戏信息
    self.message_name, self.message_proto = self.recv_request()
    # 接收游戏判断
    if self.message_name == "FightStartReq":
        self.rsp = FightStartRsp()

    elif self.message_name == "StepFrameReq":
        self.rsp = StepFrameRsp()
        # 解析来自游戏核心的步帧请求
        self._parse_frame_info(self.message_proto) 
        #self._print_debug_log(freq=1)

    elif self.message_name == "FightOverReq":
        self.rsp = FightOverRsp()
        gameover_state = self.message_proto.gameover_state
        self.game_over = True
        if gameover_state == 1:
            self.game_status = NC_CONFIG["game_status"]["win"]
        elif gameover_state == 2:
            self.game_status = NC_CONFIG["game_status"]["fail"]
        elif gameover_state == 4:
            self.game_status = NC_CONFIG["game_status"]["error"]
        #self.send_response(self.rsp)
    
    else:
        self.logger.warning("警告：接收消息失败")
    
    self.start_point = time.time()
    self.monster_hp_total.append(self.monster_hp)
    self.obs = self.agent_loc_np 
    # 状态归一化 -> [-1, 1]
    self.obs = self.obs - np.mean(self.obs)
    self.obs = self.obs / np.max(np.abs(self.obs))
    
    return self.get_obs(), self.get_state()
```      

获取第一帧的信息，获取gamecore返回的三个信息，即：
```python
self.hero_hp: {'1': 16772, '2': 5706, '3': 8409, '4': 8743, '5': 9885}  
self.monster_hp: 30000 (某个整数值)   
self.agent_loc: [[   842   2956  智能体血量 14501  -3754 怪物血量]   
                 [   861  -1095  智能体血量 14501  -3754 怪物血量]   
                 [   769  -4468  智能体血量 14501  -3754 怪物血量]         
                 [   639  -7465  智能体血量 14501  -3754 怪物血量]         
                 [   515 -10797  智能体血量 14501  -3754 怪物血量]]       
np.array()   
``` 

### 交互推进函数 step()
交互推导函数负责智能体与仿真的逐帧推导。
```python
def step(self, _actions, if_test=False):
    """返回奖励、终止状态、信息
    
    params:
        _actions: 智能体动作列表
        if_test: 是否为测试模式
        
    return:
        返回当前帧的奖励、终止状态、信息
    """
    
    if th.is_tensor(_actions):
        #actions = _actions.cpu.numpy()
        actions = [int(a) for a in _actions]
    else:
        actions = _actions
        
    self._episode_steps+=1
    print(f'当前帧：{self._episode_steps}，已执行的动作：{_actions}')
    ti = time.time() - self.start_point
    self.step_time_cost+=ti
    self.start_point = time.time()
    # 记录发送请求的数量
    self.timestep+=1
    # 每帧训练，首先给出对方动作，看对方反馈，因为reset()已经得到了初始信息，现在做的动作相当于基于初始状态做的动作
    if self.message_name == "FightStartReq":
        self.send_response(self.rsp)

    elif self.message_name == "StepFrameReq":
        # 从智能体获取动作
        cmd_list, stop_game = self.act_2_cmd(actions) # 需要重新编辑

        if stop_game:
            self.game_over = True
            self.game_status = NC_CONFIG["game_status"]["error"]
            self.rsp.gameover_ai_server = 1
            self.controller.stop_game()
            self.logger.info("向游戏核心发送游戏结束请求")

        # 发送动作命令
        # 发送步帧响应
        self.rsp.cmd_list.extend(cmd_list)
        self.send_response(self.rsp)

    elif self.message_name == "FightOverReq":
        self.send_response(self.rsp)
        
    # 初始化奖励
    reward = 0.
    terminated = False
    """
    两个终止条件：
        1. 如果运行步数超过episode_limit，这应该是终止条件，并判断游戏状态
        2. 运行步数小于episode_limit，但已经获胜
    """
    if (self._episode_steps >= self.episode_limit) or \
        ((self._episode_steps < self.episode_limit) and self.game_status==1):
        terminated = True 
        self.game_over = True
        if (self._episode_steps >= self.episode_limit):
            self.game_status = NC_CONFIG["game_status"]["overtime"]

    if not self.game_over:
        ti = time.time() - self.start_point
        self.step_time_cost+=ti 
        self.message_name, self.message_proto = self.recv_request()
        #print('接收到下一个状态！！！！！')
        if self.message_name == "FightStartReq":
            self.rsp = FightStartRsp()
            #self.send_response(self.rsp)

        elif self.message_name == "StepFrameReq":
            self.rsp = StepFrameRsp()
            # 解析来自游戏核心的步帧请求
            self._parse_frame_info(self.message_proto)
            self._print_debug_log(freq=1)

        elif self.message_name == "FightOverReq":
            self.rsp = FightOverRsp()
            gameover_state = self.message_proto.gameover_state
            self.game_over = True
            
            if gameover_state == 1:
                self.game_status = NC_CONFIG["game_status"]["win"]
            elif gameover_state == 2:
                self.game_status = NC_CONFIG["game_status"]["fail"]
            elif gameover_state == 4:
                self.game_status = NC_CONFIG["game_status"]["error"]

        self.start_point = time.time()

        self.monster_hp_total.append(self.monster_hp)
        
        self.obs = self.agent_loc_np 
        # 观测归一化 -> [-1,1]
        self.obs = self.obs - np.mean(self.obs)
        self.obs = self.obs / np.max(np.abs(self.obs))
        reward = self.get_curr_reward()
        terminated = False

    else:
        avg_time = self.step_time_cost / self.timestep
        self.logger.info(f"******* 游戏结束，状态为 {self.game_status} *******")
        self.logger.info("0: 错误, 1: 胜利, 2: 失败, 3: 超时")
        self.logger.info(
            f"帧号 = [{self.frame_no}], 步数 = [{self.timestep}], 平均时间 = [{avg_time}]")
        
        self.controller.stop_game() 
        self.battles_game+=10
        
        if self.game_status == 1:
            self.info_re['battle_won'] = True
        elif self.game_status == 2 or 3:
            self.info_re['battle_won'] = False
            
        self.info_re['monster_last_hp'] = self.monster_hp_total[-1]
        
    return reward, terminated, self.info_re
```  

### 计算奖励函数
```python
def get_curr_reward(self):
    """
    基于暴君血量返回奖励
    """
    return (self.monster_hp_total[-1] - self.monster_hp_total[-2]) * -0.01
```  

### 其他函数
#### 计算移动后的坐标
```python
def position_change(self, agent_id, act):
    '''用于计算移动后的位置坐标
    '''
    # self.agent_loc_np是上次保存的状态信息
    dst_pos=VInt3()
    last_loc_x = self.agent_loc_np[agent_id][0] # 获取上一轮的x坐标位置
    last_loc_z = self.agent_loc_np[agent_id][1] # 获取上一轮的z坐标位置
    if act==0:
        dst_pos.x = last_loc_x + self.pos_delta 
        dst_pos.z = last_loc_z
        dst_pos.y = 100

    elif act==1:
        dst_pos.x = last_loc_x - self.pos_delta 
        dst_pos.z = last_loc_z
        dst_pos.y = 100

    elif act==2:
        dst_pos.x = last_loc_x 
        dst_pos.z = last_loc_z + self.pos_delta
        dst_pos.y = 100

    elif act==3:
        dst_pos.x = last_loc_x
        dst_pos.z = last_loc_z - self.pos_delta
        dst_pos.y = 100
    
    # """细化移动动作：左上、右上、右下、左下"""
    elif act==9:
        # 左上
        dst_pos.x = last_loc_x + self.pos_delta
        dst_pos.z = last_loc_z - self.pos_delta
        dst_pos.y = 100
        
    elif act==10:
        # 右上
        dst_pos.x = last_loc_x + self.pos_delta
        dst_pos.z = last_loc_z + self.pos_delta
        dst_pos.y = 100
        
    elif act==11:
        # 右下
        dst_pos.x = last_loc_x - self.pos_delta
        dst_pos.z = last_loc_z + self.pos_delta
        dst_pos.y = 100
        
    elif act==12:
        # 左下
        dst_pos.x = last_loc_x - self.pos_delta
        dst_pos.z = last_loc_z - self.pos_delta
        dst_pos.y = 100
        
    # 边界处理
    if dst_pos.x > 28700:
        dst_pos.x = 28700
    elif dst_pos.x < -28700:
        dst_pos.x = -28700
    
    if dst_pos.z > 9000:
        dst_pos.z = 9000
    elif dst_pos.z < -16800:
        dst_pos.z = -16800
    
    return dst_pos
```  
注意：
通过记录上一帧的坐标并加上当前帧的偏移量，获得最新坐标
- act=0 是沿x向上移动
- act=1 是沿x向下移动
- act=2 是沿z向上移动
- act=3 是沿z向下移动
- act=9 是沿左上方向移动
- act=10 是沿右上方向移动
- act=11 是沿右下方向移动
- act=12 是沿左下方向移动

