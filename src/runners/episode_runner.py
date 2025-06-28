###
# 这个程序是改进版的 episode_runner.py，增加了复杂的奖励塑形
###

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import os
import json

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        print(self.batch_size)
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.episode_total = 0
        
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, if_test=None, cur_time=None):
        self.batch = self.new_batch()
        if if_test==None:
            self.env.reset(args=self.args, cur_time=cur_time)
        else:
            self.env.reset(if_test=if_test,args=self.args, cur_time=cur_time)

        self.t = 0

    def run(self, test_mode=False, cur_time=None):
        monster_hp_json_file_path = os.path.join('./results/sacred/hok/monster_last_hp/', f'{cur_time}_monster_lasthp_{self.args.name}.txt')
        if self.args.env=='hok':
            self.reset(if_test=test_mode, cur_time=cur_time)
        else:
            self.reset(cur_time=cur_time)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=slice(self.t, self.t + 1))

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
           
            #actions = actions[0] # for ippo
            # Fix memory leak
            cpu_actions = actions[0].to("cpu").numpy()

            # 这是跟环境交互的最重要的一步
            reward, terminated, env_info = self.env.step(actions[0], if_test=test_mode)
            original_reward = reward

            # 2. 攻击激励 (来源于PDF的思路) - 优化超参数
            NORMAL_ATTACK_IDS = [4]
            SKILL_ATTACK_IDS = [5, 6, 7, 8]
            
            num_normal_attacks = 0 # k
            num_skill_attacks = 0  # m
            num_lazy_agents = 0    # 统计"懒惰"的智能体

            # actions[0] 是一个包含所有智能体动作ID的张量
            for action in actions[0]:
                action_id = action.item()
                if action_id in NORMAL_ATTACK_IDS:
                    num_normal_attacks += 1
                elif action_id in SKILL_ATTACK_IDS:
                    num_skill_attacks += 1
                else: # 既不普攻也不技能，就算"懒惰"
                    num_lazy_agents += 1

            # 根据PDF第12页的描述，设定奖励 shaping 的参数 - 优化后的超参数
            # 增加攻击激励强度，减少懒惰智能体现象
            alpha = 1.3  # 从1.1增加到1.3，增强普攻激励
            beta = 1.5   # 从1.2增加到1.5，增强技能攻击激励
            
            # 计算带有攻击激励的奖励
            attack_shaped_reward = original_reward * (alpha**num_normal_attacks) * (beta**num_skill_attacks)

            # --- 初始化最终奖励 ---
            final_shaped_reward = attack_shaped_reward
            
            # --- 3. 生存激励 ---
            # 只要游戏没结束，就给予微小的生存奖励，鼓励英雄存活
            SURVIVAL_BONUS = 0.005  # 从0.01减少到0.005，避免过度鼓励生存而忽略攻击
            if not terminated:
                final_shaped_reward += SURVIVAL_BONUS

            # --- 4. 协作激励 与 懒惰惩罚 ---
            # 获取所有英雄的观测信息，其中包含了位置坐标
            all_obs = self.env.get_obs()
            
            # 懒惰惩罚: 每有一个英雄"挂机"，就给予一个较大的惩罚 - 增强惩罚
            LAZY_PENALTY = -0.05  # 从-0.02增加到-0.05，更严厉惩罚懒惰行为
            final_shaped_reward += num_lazy_agents * LAZY_PENALTY

            # 协作激励: 鼓励辅助(Agent 0, 庄周)靠近射手(Agent 1, 狄仁杰)
            try:
                # 假设 Agent 0 是庄周, Agent 1 是狄仁杰
                zhuangzhou_obs = all_obs[0]
                direnjie_obs = all_obs[1]
                
                # 观测的前两位是 x, z 坐标
                zhuangzhou_pos = zhuangzhou_obs[:2]
                direnjie_pos = direnjie_obs[:2]
                
                # 计算欧氏距离
                distance = np.linalg.norm(zhuangzhou_pos - direnjie_pos)
                
                COOP_DISTANCE_THRESHOLD = 1500  # 从2000减少到1500，鼓励更紧密协作
                COOP_BONUS = 0.08  # 从0.05增加到0.08，增强协作奖励

                if distance < COOP_DISTANCE_THRESHOLD:
                    # 距离够近，给予协作奖励
                    final_shaped_reward += COOP_BONUS
                    
                # 额外奖励：如果庄周在协作距离内且进行攻击
                if distance < COOP_DISTANCE_THRESHOLD:
                    zhuangzhou_action = actions[0][0].item()  # 庄周的动作
                    if zhuangzhou_action in NORMAL_ATTACK_IDS + SKILL_ATTACK_IDS:
                        ACTIVE_COOP_BONUS = 0.1  # 新增：积极协作奖励
                        final_shaped_reward += ACTIVE_COOP_BONUS
                        
            except IndexError:
                # 以防万一有英雄阵亡，导致列表越界
                pass
                
            # --- 5. 新增：团队攻击协调奖励 ---
            # 如果多个智能体同时攻击，给予额外奖励
            if num_normal_attacks + num_skill_attacks >= 3:  # 至少3个智能体攻击
                TEAM_ATTACK_BONUS = 0.12  # 团队攻击奖励
                final_shaped_reward += TEAM_ATTACK_BONUS
            elif num_normal_attacks + num_skill_attacks >= 2:  # 至少2个智能体攻击
                TEAM_ATTACK_BONUS = 0.06  # 较小的团队攻击奖励
                final_shaped_reward += TEAM_ATTACK_BONUS

            episode_return += reward
            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(final_shaped_reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=slice(self.t, self.t + 1))
            self.t += 1
            #print(f'self.t {self.t}, terminated {terminated}')

            # 加入暴龙血量的文件保存
            if env_info['monster_last_hp'] is not None:
                '''此时到达了最后一帧，可以拿到当前帧的大龙血量了 -> [1, 2, ...]'''
                # 检查目标文件夹是否存在，如果不存在则创建
                if not os.path.exists(os.path.dirname(monster_hp_json_file_path)):
                    os.makedirs(os.path.dirname(monster_hp_json_file_path))
                # 将数据存储到与当前文件夹相同的文件中，如果没有这个文件，则先创建
                if not os.path.exists(monster_hp_json_file_path):
                    with open(monster_hp_json_file_path, 'w') as file:
                        file.write('[]')
                    
                    with open(monster_hp_json_file_path, 'r') as file:
                        content = file.read()
                        numbers = eval(content)
                        numbers.append(env_info['monster_last_hp'])
                    with open(monster_hp_json_file_path, 'w') as file:
                        file.write(str(numbers))
                # 如果有了，则采用追加格式
                else:
                    with open(monster_hp_json_file_path, 'r') as file:
                        content = file.read()
                        numbers = eval(content)
                        numbers.append(env_info['monster_last_hp'])
                    with open(monster_hp_json_file_path, 'w') as file:
                        file.write(str(numbers))
            
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        } # last_data应该没有作用，这个是充数的
        self.batch.update(last_data, ts=slice(self.t, self.t + 1))

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=slice(self.t, self.t + 1))
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.episode_total += 1

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, env_info

    def _log(self, returns, stats, prefix):
        # stats: {'battle_won': 0, 'n_episodes': 1, 'ep_length': 45} battle_won从哪里来的
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
