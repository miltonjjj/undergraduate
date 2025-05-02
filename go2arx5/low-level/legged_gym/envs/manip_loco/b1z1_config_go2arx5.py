# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# 导入所需的模块
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import torch
# 定义B1Z1RoughCfg配置类，继承自LeggedRobotCfg
class B1Z1RoughCfg( LeggedRobotCfg ):
    class goal_ee:# 机器人末端执行器的目标配置
        num_commands = 3  # 命令数量
        traj_time = [1, 3]  # 轨迹时间
        hold_time = [0.5, 2]  # 保持时间
        collision_upper_limits = [0.3, 0.18, -0.05]  # 红色碰撞空间的上限
        collision_lower_limits = [-0.3, -0.18, -0.46]  # 碰撞空间的下限
        underground_limit = 0.  # 地下限制
        num_collision_check_samples = 10  # 碰撞检查样本数
        command_mode = 'sphere'  # 命令模式，使用球形
        arm_induced_pitch = 0.38  # 末端执行器的目标俯仰角度（负的目标俯仰角）
        # 定义末端执行器的目标球体中心
        class sphere_center:
            x_offset = 0.04  # x方向偏移
            y_offset = 0.0  # y方向偏移
            z_invariant_offset = 0.46  # z方向偏移
        # 定义末端执行器的运动范围
        class ranges:
            # 初始位置范围的起始值，包含x、y、z三个坐标，单位是m和rad
            init_pos_start = [0.5, np.pi/8, -0.2]
            # 初始位置范围的结束值，包含x、y、z三个坐标，单位是m和rad
            init_pos_end = [0.4, 0., -0.3]
            # x方向的位置范围，单位是m
            pos_l = [0.4, 0.95]
            # 俯仰角度（pitch）的范围，单位是rad
            pos_p = [-1 * np.pi / 5, 1 * np.pi / 4]
            # 偏航角度（yaw）的范围，单位是rad
            pos_y = [-1.2, 1.2]
            
            # 旋转的范围，r表示滚转角（roll），p表示俯仰角（pitch），y表示偏航角（yaw）
            delta_orn_r = [-0.5, 0.5]  # 滚转角的范围，单位是rad
            delta_orn_p = [-0.5, 0.5]  # 俯仰角的范围，单位是rad
            delta_orn_y = [-0.5, 0.5]  # 偏航角的范围，单位是rad
            # 末端执行器跟踪奖励的最终值，表示一个期望的奖励值
            final_tracking_ee_reward = 0.55

        # 末端执行器的错误尺度（误差范围） - 用于计算误差的缩放因子
        sphere_error_scale = [1, 1, 1]  # 对应x、y、z方向的误差尺度，初始为1，即没有缩放
        # 末端执行器的姿态误差尺度 - 用于计算姿态误差的缩放因子
        orn_error_scale = [1, 1, 1]  # 对应滚转、俯仰和偏航角的误差尺度，初始为1，即没有缩放


 # 定义噪声配置
    class noise:
        add_noise = False  # 是否添加噪声
        noise_level = 1.0  # 噪声级别
        class noise_scales:# 噪声的具体缩放因子
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1  

    class commands:# 定义命令的配置
        curriculum = True  # 是否使用渐进式学习
        num_commands = 3  # 命令数量
        resampling_time = 3.0  # 命令更换时间

        lin_vel_x_schedule = [0, 0.5]  # x轴线速度调度
        ang_vel_yaw_schedule = [0, 1]  # 偏航角速度调度
        tracking_ang_vel_yaw_schedule = [0, 1]  # 偏航角度跟踪调度
        #这些调度参数设定了运动指令在训练中的变化区间，用于控制机器人期望达到的运动速度和旋转速度。
        ang_vel_yaw_clip = 0.5  # 偏航角速度的限制
        lin_vel_x_clip = 0.2  # x轴线速度的限制

        class ranges: # 线速度和角速度的范围
            lin_vel_x = [-0.6, 0.6]  # x轴线速度范围 [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # 偏航角速度范围 [rad/s]

    class normalization: # 定义标准化配置
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.0  # 限制观测值
        clip_actions = 100.0  # 限制动作值

    class env:# 定义环境的配置
        num_envs = 3072  # 环境数量
        num_actions = 12 + 6  # 动作数量
        num_torques = 12 + 6  # 扭矩数量
        action_delay = 3  # 动作延迟，-1为无延迟
        num_gripper_joints = 2  # 夹爪关节数量
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3  # 本体状态信息的数量
        num_priv = 5 + 1 + 12  # 特权信息数量
        history_len = 10  # 历史状态长度
        num_observations = num_proprio * (history_len+1) + num_priv  # 总的观测值数量
        num_privileged_obs = None  # 特权观测值
        send_timeouts = True  # 是否发送超时信息
        episode_length_s = 10  # 每集的时长（秒）
        reorder_dofs = True  # 是否重新排列自由度
        teleop_mode = False  # 是否启用远程控制模式
        record_video = False  # 是否录制视频
        stand_by = False  # 是否处于待机状态
        observe_gait_commands = False  # 是否观察步态命令
        frequencies = 2  # 更新频率

    class init_state( LeggedRobotCfg.init_state ): # 初始化状态配置
        pos = [0.0, 0.0, 0.34]  # 初始位置 [x, y, z]
        default_joint_angles = {  # 四足和机械臂的初始关节角度
            'FL_hip_joint': 0.1,  # 前左髋关节 [rad]
            'RL_hip_joint': 0.1,  # 后左髋关节 [rad]
            'FR_hip_joint': -0.1,  # 前右髋关节 [rad]
            'RR_hip_joint': -0.1,  # 后右髋关节 [rad]

            'FL_thigh_joint': 0.8,  # 前左大腿关节 [rad]
            'RL_thigh_joint': 1.,  # 后左大腿关节 [rad]
            'FR_thigh_joint': 0.8,  # 前右大腿关节 [rad]
            'RR_thigh_joint': 1.,  # 后右大腿关节 [rad]

            'FL_calf_joint': -1.5,  # 前左小腿关节 [rad]
            'RL_calf_joint': -1.5,  # 后左小腿关节 [rad]
            'FR_calf_joint': -1.5,  # 前右小腿关节 [rad]
            'RR_calf_joint': -1.5,  # 后右小腿关节 [rad]

            # 机械臂的初始关节角度
            "zarx_j1": 0.0,
            "zarx_j2": 0.1,
            "zarx_j3": 0.1,
            "zarx_j4": 0.0,
            "zarx_j5": 0.0,
            "zarx_j6": 0.0,
            "zarx_j7": 0.0,
            "zarx_j8": 0.0,
        }
        rand_yaw_range = np.pi/2  # 随机偏航角范围
        origin_perturb_range = 0.5  # 原点扰动范围
        init_vel_perturb_range = 0.1  # 初始化速度扰动范围

    # 控制配置
    class control(LeggedRobotCfg.control):
        stiffness = {  # 各关节的刚度
            'joint': 35.,
            "zarx": 50.,
            "zarx_j1": 40.,
            "zarx_j2": 70.,
            "zarx_j3": 70.,
            "zarx_j4": 25.,
            "zarx_j5": 25.,
            "zarx_j6": 25.,
            "zarx_j7": 50.,
            "zarx_j8": 50.,          
        }  # [N*m/rad] 
        damping = {  # 各关节的阻尼
            'joint': 1.,
            "zarx": 20.,
            "zarx_j1": 3.,
            "zarx_j2": 15.,
            "zarx_j3": 15.,
            "zarx_j4": 2.,
            "zarx_j5": 2.,
            "zarx_j6": 2.,
            "zarx_j7": 20.,
            "zarx_j8": 20.,
        }  # [N*m*s/rad]

        adaptive_arm_gains = False  # 是否使用自适应机械臂增益
        # 动作的缩放因子
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        decimation = 4  # 动作更新的减法次数
        torque_supervision = False  # 扭矩监督

    # 机器人资源的配置
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/arx5go2.urdf'  # 机器人资源文件路径
        foot_name = "foot"  # 足部名称
        gripper_name = "zarx_body7"  # 夹爪名称
        penalize_contacts_on = ["thigh",  "calf", "trunk"]  # 关节接触惩罚
        terminate_after_contacts_on = []  # 结束后接触
        self_collisions = 0  # 是否启用自碰撞检查
        flip_visual_attachments = False  # 是否翻转视觉附件
        collapse_fixed_joints = True  # 是否合并固定关节
        fix_base_link = False  # 是否固定基座链接
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
    
    class arm:
        init_target_ee_base = [0.0, 0.0, 0.0]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    # 随机化配置
    class domain_rand:
        observe_priv = True  # 是否观察特权信息
        randomize_friction = True  # 是否随机化摩擦
        friction_range = [0.05, 4.5]  # 摩擦范围
        randomize_base_mass = True  # 是否随机化基础质量
        added_mass_range = [-0.5,0.5]  # 添加质量的范围
        randomize_base_com = True  # 是否随机化基础质心
        added_com_range_x = [-0.05, 0.05]  # 质心x方向范围
        added_com_range_y = [-0.05, 0.05]  # 质心y方向范围 
        added_com_range_z = [-0.05, 0.05]  # 质心z方向范围
        randomize_motor = True  # 是否随机化电机
        leg_motor_strength_range = [0.9, 1.1]  # 腿部电机的强度范围
        arm_motor_strength_range = [0.9, 1.1]  # 机械臂电机的强度范围
        randomize_gripper_mass = True  # 是否随机化夹爪的质量
        gripper_added_mass_range = [0, 0.1]  # 夹爪添加质量的范围
        push_robots = True  # 是否推动机器人
        push_interval_s = 8  # 推动机器人间隔时间
        max_push_vel_xy = 0.5  # 最大推动速度

    # 奖励配置
    class rewards:
        reward_container_name = "maniploco_rewards"  # 奖励容器名称

        # -------常规参数--------
        only_positive_rewards = False  # 是否仅使用正奖励
        tracking_sigma = 0.2  # 跟踪奖励的sigma
        tracking_ee_sigma = 1  # 末端执行器的sigma
        soft_dof_pos_limit = 1.  # 自由度位置限制
        soft_dof_vel_limit = 1.  # 自由度速度限制
        soft_torque_limit = 0.4  # 扭矩限制
        base_height_target = 0.34  # 基座高度目标
        max_contact_force = 40.  # 最大接触力

        # -------步态控制参数---------
        gait_vel_sigma = 0.5  # 步态速度的sigma，归一化因子，sigma越小惩罚越大
        gait_force_sigma = 0.5  # 步态力的sigma，同理系数越小惩罚越大
        kappa_gait_probs = 0.07  # 步态的概率
        feet_height_target = 0.3  # 足部高度目标
        #足部高度需要训练之后再看结果

        feet_aritime_allfeet = False  # 是否所有足部都处于空中
        feet_height_allfeet = False  # 是否所有足部都在同一高度

        # 奖励比例
        class scales:
            tracking_contacts_shaped_force = -3.0  # 接触奖励（力量） -2
            tracking_contacts_shaped_vel = -1.0  # 接触奖励（速度） -2
            feet_air_time = 2.0  # 足部空中时间奖励 2
            feet_height = 1.0  # 足部高度奖励 1

            tracking_lin_vel_max = 0.5  # 最大线速度奖励
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0.
            tracking_ang_vel = 0.5  # 角速度奖励

            delta_torques = -1.0e-7/4.0
            work = 0
            energy_square = 0.0
            torques = -2.5e-4 
            stand_still = 1.0 
            walking_dof = 1.5
            dof_default_pos = 0.0
            dof_error = 0.0 
            alive = 1.0
            lin_vel_z = -1.5
            roll = -2

            # --------共通奖励----------
            ang_vel_xy = -0.05  # xy方向角速度奖励
            dof_acc = -7.5e-7  # 自由度加速度奖励
            collision = -2.  # 碰撞惩罚
            action_rate = -0.015  # 动作速率惩罚
            dof_pos_limits = -10.0  # 自由度位置限制惩罚
            delta_torques = -1.0e-7
            hip_pos = -0.3
            work = -0.003
            feet_jerk = -0.0002
            feet_drag = -0.03
            feet_contact_forces = -0.001
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            base_height = -5.0
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
            penalty_lin_vel_y = 0.

        # 奖励配置的具体数值
        class arm_scales:
            arm_termination = None  # 机械臂终止奖励
            tracking_ee_sphere = 0.  # 末端执行器球体跟踪奖励
            tracking_ee_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None

    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    # 终止条件配置
    class termination:
        r_threshold = 0.8  # x轴位置阈值
        p_threshold = 0.8  # y轴位置阈值
        z_threshold = 0.1  # z轴位置阈值

    # 地形配置
    class terrain:
        mesh_type = 'trimesh'  # 地形网格类型
        hf2mesh_method = "fast"  # 快速网格转换方法
        max_error = 0.1  # 最大误差
        horizontal_scale = 0.05  # 水平尺度
        vertical_scale = 0.005  # 垂直尺度
        border_size = 25  # 边界大小
        height = [0.00, 0.1]  # 高度范围
        gap_size = [0.02, 0.1]  # 间隙大小
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 # number of terrain cols (types)


        # 地形的不同类型及其比例
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0.,
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())  # 地形类型的比例
        # 对于trimesh类型的地形：
        slope_treshold = None  # 大于该阈值的坡度将被矫正为垂直面
        origin_zero_z = False  # 是否将原点的z值设为零

# B1Z1RoughCfgPPO类继承自LeggedRobotCfgPPO类，定义了PPO训练的具体配置
class B1Z1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1  # 随机数种子
    runner_class_name = 'OnPolicyRunner'  # 运行器的类名
    
    # 定义策略的配置
    class policy:
        continue_from_last_std = True  # 是否从上次的标准偏差继续
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]  # 初始标准差
        actor_hidden_dims = [128]  # actor网络的隐藏层维度
        critic_hidden_dims = [128]  # critic网络的隐藏层维度
        activation = 'elu'  # 激活函数类型
        output_tanh = False  # 是否输出tanh激活值

        # 关节控制的头部隐藏层维度
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        # 特权信息的编码维度
        priv_encoder_dims = [64, 20]

        # 腿部和机械臂的动作数量
        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains  # 是否使用自适应机械臂增益
        adaptive_arm_gains_scale = 10.0  # 自适应增益的缩放因子
        
    # 算法配置
    class algorithm:
        value_loss_coef = 1.0  # 值函数损失的权重
        use_clipped_value_loss = True  # 是否使用剪切的值函数损失
        clip_param = 0.2  # 剪切参数
        entropy_coef = 0.0  # 熵的系数
        num_learning_epochs = 5  # 每次更新时学习的轮数
        num_mini_batches = 4  # 每个小批次的环境数量
        learning_rate = 2e-4  # 学习率
        schedule = 'fixed'  # 学习率调度方式
        gamma = 0.99  # 折扣因子
        lam = 0.95  # 优先级回放的lambda参数
        desired_kl = None  # 目标KL散度
        max_grad_norm = 1.0  # 最大梯度范数
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]  # 最小策略标准差

        mixing_schedule = [1.0, 0, 3000]  # 混合调度
        torque_supervision = B1Z1RoughCfg.control.torque_supervision  # 扭矩监督
        torque_supervision_schedule = [0.0, 1000, 1000]  # 扭矩监督调度
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains  # 是否使用自适应机械臂增益
        dagger_update_freq = 20  # dagger算法更新频率
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]  # 特权正则化系数调度
        
    # 训练过程中的运行配置
    class runner:
        policy_class_name = 'ActorCritic'  # 策略类名
        algorithm_class_name = 'PPO'  # 算法类名
        num_steps_per_env = 24  # 每个环境的步数
        max_iterations = 36000  # 最大训练迭代次数
        save_interval = 4000  # 保存模型的间隔次数
        experiment_name = 'b1z1_v2'  # 实验名称
        run_name = ''  # 运行名称
        
        # 加载和恢复训练
        resume = False  # 是否恢复训练
        load_run = -1  # 恢复的运行编号
        checkpoint = -1  # 恢复的检查点编号
        resume_path = None  # 恢复路径
        #版本：0317v3            

