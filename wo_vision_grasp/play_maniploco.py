from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch
import os
import torch.nn as nn

from utils.config import load_cfg, get_params, copy_cfg
import utils.wrapper as wrapper
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import sys
from types import SimpleNamespace

from envs import *
from path_planner import GridMap, PathPlanner
def create_env(cfg, args):
    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = args.small_value_set_zero
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)
    robot_start_pose = (-2.00, 1.50, 0.55)
    if args.eval:
        robot_start_pose = (-0.85, 0, 0.55)
    
    # 修复环境类的引用方式
    if args.task == "B1Z1ManipLoco":
        from envs.b1z1_maniploco import B1Z1ManipLoco
        _env = B1Z1ManipLoco(cfg=cfg, sim_device=args.sim_device, rl_device=args.rl_device,
                           graphics_device_id=args.graphics_device_id, headless=args.headless,
                           use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands, 
                           no_feature=args.no_feature, mask_arm=args.mask_arm, pitch_control=args.pitch_control,
                           rand_control=args.rand_control, arm_delay=args.arm_delay, robot_start_pose=robot_start_pose,
                           rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip, 
                           stop_pick=args.stop_pick, table_height=args.table_height, eval=args.eval)
    else:
        raise ValueError(f"未知任务类型: {args.task}")
    
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

# 添加可视化函数
def visualize_path(grid_map, path, robot_pos, target_pos, save_path=None):
    
    plt.figure(figsize=(10, 10))
    
    # 计算地图范围
    map_width = grid_map.grid_width * grid_map.resolution
    map_height = grid_map.grid_height * grid_map.resolution
    extent = [-map_width/2, map_width/2, -map_height/2, map_height/2]
    
    # 绘制网格地图（障碍物为白色，可通行区域为黑色）
    #plt.imshow(grid_map.grid.T, origin='lower', extent=extent, cmap='binary')
    cmap = ListedColormap(['white', 'black', 'gray'])
    plt.imshow(grid_map.grid.T, origin='lower', extent=extent, cmap=cmap)
    # 如果有路径，绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2, label='planned path')
    
    # 绘制机器人初始位置和目标位置
    plt.plot(robot_pos[0], robot_pos[1], 'go', markersize=10, label='robot')
    plt.plot(target_pos[0], target_pos[1], 'bo', markersize=10, label='goal')
    
    # 绘制当前环境中的障碍物（使用不同颜色标识）
    # 这里只是示例，您可能需要根据实际情况修改
    plt.scatter(0.0, 0.0, color='yellow', s=100, alpha=0.7, label='obstacles')
    
    # 设置图表属性
    plt.title('path planner')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存至: {save_path}")
    
    # 显示图像
    #plt.show()


#主函数
def main():
    args = get_params()
    args.eval = is_eval
    args.wandb = args.wandb and (not args.eval) and (not args.debug)

    # 使用maniploco.yaml配置文件
    cfg_file = "b1z1_maniploco.yaml"  # 使用已存在的配置文件作为基础
    file_path = "data/cfg/" + cfg_file
        
    cfg = load_cfg(file_path)
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob

    stats = SimpleNamespace(successful=0, total=0)

    # 创建一个函数来执行路径规划和初始化跟踪器,返回实例化的PathTracker
    def plan_new_path(env, grid_map, stats):
       
        robot_state = env._robot_root_states
        robot_init_pos = (
            robot_state[0, 0].cpu().item(), 
            robot_state[0, 1].cpu().item()
        )
        
        # 获取物体的当前位置
        cube_pos_x = env._cube_root_states[0, 0].cpu().item()
        cube_pos_y = env._cube_root_states[0, 1].cpu().item()
        
        # 障碍物中心位置和尺寸
        obstacle_center = (0.0, 0.0)
        obstacle_size = 0.6
        obstacle_radius = obstacle_size / 2.0 *1.4
        safe_distance = 0.65  # 机器人与障碍物的安全距离
        
        # 计算从障碍物中心到物体的方向向量
        obstacle_to_cube_x = cube_pos_x - obstacle_center[0]
        obstacle_to_cube_y = cube_pos_y - obstacle_center[1]
        
        # 计算障碍物中心到物体的距离
        obstacle_to_cube_distance = math.sqrt(obstacle_to_cube_x**2 + obstacle_to_cube_y**2)
        
        # 归一化方向向量
        if obstacle_to_cube_distance > 0:
            direction_x = obstacle_to_cube_x / obstacle_to_cube_distance
            direction_y = obstacle_to_cube_y / obstacle_to_cube_distance
        else:
            # 如果物体在障碍物中心，默认取一个方向
            direction_x = 1.0
            direction_y = 0.0
        
        # 计算目标位置：障碍物边缘外一定距离的位置，朝向物体方向
        target_distance = obstacle_radius + safe_distance
        target_pos = (
            obstacle_center[0] + direction_x * target_distance,
            obstacle_center[1] + direction_y * target_distance
        )

        path_planner = PathPlanner(grid_map)
        path = path_planner.plan_path(robot_init_pos, target_pos)
        
        if not path:
            print("路径规划失败")
            sys.exit()
        
        visualize_path(grid_map, path, robot_init_pos, target_pos, save_path="path_planning_result.png")
        
        path_tracker = PathTracker(path, env, stats)
        return path_tracker
    
    
    # 定义路径跟踪控制器
    class PathTracker:
        def __init__(self, path, env, stats):
            self.path = path
            self.stats = stats
            #self.actions = actions
            self.current_waypoint_idx = 0
            self.env = env
            self.max_linear_speed = 1.#0.5  # 最大线速度
            self.max_angular_speed = 1.#0.5  # 最大角速度

            # 抓取状态管理
            self.grasping_state = "navigating"  # 初始状态：导航中
            self.grasp_timer = 0  # 用于计时各阶段持续时间
            #self.pitch_angle = 0.0  # 机器人的pitch角度
            #self.target_pitch = 0.3  # 目标pitch角度(弧度)，前低后高

            # 抓取尝试计数
            self.grasp_attempt_count = 0  # 记录抓取尝试次数
            self.max_grasp_attempts = 3   # 最大尝试次数
    
            # 每个阶段的持续时间(步数)
            self.align_time = 30
            self.arm_approach_time = 40
            self.grasp_time = 5
            self.lift_time = 10  

            # 状态变量初始化（放在 __init__ 或 reset 中）
            self.position_error_integral = torch.zeros(3, device=env.device)
            self.prev_position_error = torch.zeros(3, device=env.device)
            self.max_position_integral = 0.5

            self.position_kp = 0.1
            self.position_ki = 0.08
            self.position_kd = 0.01

            self.orientation_error_integral = torch.zeros(3, device=env.device)
            self.prev_orientation_error = torch.zeros(3, device=env.device)
            self.max_orientation_integral = 0.2

            self.orientation_kp = 0.1
            self.orientation_ki = 0.02
            self.orientation_kd = 0.005

            self.flag=0

        #navigating和aligning阶段的动作命令，返回速度命令
        def get_action(self, current_pos, current_heading):
             # 导航状态处理
            if self.grasping_state == "navigating":
                # 检查是否已经足够接近最终目标点
                final_dx = self.path[-1][0] - current_pos[0]
                final_dy = self.path[-1][1] - current_pos[1]
                final_distance = math.sqrt(final_dx**2 + final_dy**2)
                
                if final_distance < 0.4:  
                    self.grasping_state = "aligning"
                    self.grasp_timer = 0
                    print(f"导航完成，到达目标位置")
                    print(f"(1)完成navigating,开始aligning")
                    return 0.0, 0.0  # 停止移动
                
                while True:
                    # 计算下一个路径点
                    if self.current_waypoint_idx + 1 >= len(self.path):
                        # 已经是最后一个路径点，使用最后一个点
                        next_waypoint = self.path[-1]
                    else:
                        next_waypoint = self.path[self.current_waypoint_idx + 1]
                    
                    # 计算到下一个路径点的向量
                    dx = next_waypoint[0] - current_pos[0] 
                    dy = next_waypoint[1] - current_pos[1]
                    
                    # 计算到下一个路径点的距离
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    # 如果足够接近当前路径点，移动到下一个路径点
                    if distance < 0.4 and self.current_waypoint_idx + 1 < len(self.path):
                        self.current_waypoint_idx += 1
                    else:
                        # 当前点不够接近或已是最后一个点，跳出循环
                        break
                # 计算目标朝向角
                target_heading = math.atan2(dy, dx)
                
                # 计算朝向误差
                heading_error = target_heading - current_heading
                # 标准化角度到 [-pi, pi]
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                    
                # PD控制器参数
                kp_linear = 1.0
                kp_angular = 2.0
                
                # 朝向误差的绝对值
                abs_heading_error = abs(heading_error)
                
                # 首先调整机器人朝向再前进
                if abs_heading_error > 0.3:  # 朝向误差超过约17度时
                    # 减小线速度，优先调整朝向
                    linear_vel = 0.0  # 当朝向误差大时，先停止前进
                    angular_vel = min(self.max_angular_speed, max(-self.max_angular_speed, kp_angular * heading_error))
                elif abs_heading_error > 0.1:  # 朝向误差较小但仍需调整
                    # 以较低的速度前进，同时继续调整朝向
                    linear_vel = min(self.max_linear_speed * 0.3, kp_linear * distance * 0.3)
                    angular_vel = min(self.max_angular_speed, max(-self.max_angular_speed, kp_angular * heading_error))
                else:
                    # 朝向基本正确，可以正常前进
                    linear_vel = min(self.max_linear_speed, kp_linear * distance)
                    angular_vel = min(self.max_angular_speed, max(-self.max_angular_speed, kp_angular * heading_error))
                   
                return linear_vel, angular_vel
            
            # 朝向调整状态处理
            elif self.grasping_state == "aligning":
                self.grasp_timer += 1
                
                # 获取物体位置
                cube_pos = self.env._cube_root_states[0, :2].cpu().numpy()
                robot_pos = np.array([current_pos[0], current_pos[1]])
                
                # 计算朝向物体的目标角度
                dx = cube_pos[0] - robot_pos[0]
                dy = cube_pos[1] - robot_pos[1]
                target_heading = math.atan2(dy, dx)
                distance_to_cube = math.sqrt(dx**2 + dy**2)
        
                # 计算朝向误差
                heading_error = target_heading - current_heading
                # 标准化角度到 [-pi, pi]
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                
                # 阶段1：先调整朝向，朝向基本正确后再靠近
                if abs(heading_error) > 0.1 and self.grasp_timer < self.align_time:
                    # 先调整朝向，不移动
                    angular_vel = min(self.max_angular_speed, max(-self.max_angular_speed, 2.0 * heading_error))
                    return 0.0, angular_vel
                
                # 阶段2：朝向已调好，检查距离
                if distance_to_cube > 0.9:  
                    # 以较低速度直线靠近物体
                    linear_vel = min(0.3, 0.6 * distance_to_cube)
                    # 同时保持正确朝向
                    angular_vel = min(self.max_angular_speed, max(-self.max_angular_speed, 1.5 * heading_error))
                    print(f"靠近物体中，当前距离: {distance_to_cube:.2f}m")
                    return linear_vel, angular_vel
                
                # 距离合适且朝向正确，可以开始抓取
                self.grasping_state = "approaching"
                self.grasp_timer = 0
                print(f"已到达适合抓取的位置，距离物体: {distance_to_cube:.2f}m")
                print(f"(2)完成aligning,开始approaching")
                return 0.0, 0.0
        
        #approaching,grasping和lifting阶段的动作命令，返回动作向量
        def get_grasping_actions(self, actions):
            """根据当前抓取状态设置完整的动作张量"""
            # 设置基本速度命令（停止移动）
            device = env.rl_device
            actions = torch.zeros(env.num_envs, env.num_actions, device=device, dtype=torch.float32)
            #计时器
            self.grasp_timer += 1

            if self.grasping_state == "approaching":
                
                 # 保持夹爪张开状态
                actions[:, 6] = 1
                actions[:, 7] = 0. 
                actions[:, 8] = 0.
                actions[:, 9] = 0.
                #if path_tracker.grasping_state == "approaching":
                    #print("approaching:grasp_timer=",path_tracker.grasp_timer)
                
                # 设置夹爪方向 - 确保夹爪朝向物体
                #actions[:, 3] = 0.0  # roll
                #actions[:, 4] = 0.1  # 稍微调整pitch，使夹爪更适合抓取
                #actions[:, 5] = 0.0  # yaw
                cube_pos = self.env._cube_root_states[0, :3].cpu()
                current_ee_pos = self.env.ee_pos[0].cpu()
                delta_pos= cube_pos - current_ee_pos 
                distance = torch.norm(delta_pos)

                if distance < 0.05:
                #if distance<0.05 and delta_pos[2]<0.02 and delta_pos[1]<0.02 and delta_pos[0]<0.02:   
                    self.grasping_state = "grasping"
                    self.grasp_timer = 0
                    print(f"(3)完成approaching,开始grasping")
                    print(f"机械臂距离物体:{distance:.3f}m,开始抓取")
                
                elif self.grasp_timer >= self.arm_approach_time:
                    self.grasping_state = "grasping"
                    self.grasp_timer = 0
                    print(f"(3)完成approaching,开始grasping")
                    print(f"approaching时间耗尽,开始抓取")

            elif self.grasping_state == "grasping":
                actions[:, 7] = 0.0 
                actions[:, 8] = 0.
                actions[:, 6] = -1#夹爪
                #if path_tracker.grasping_state == "grasping":
                    #print("grasping:grasp_timer=",path_tracker.grasp_timer)
                
                if self.grasp_timer >= self.grasp_time:
                    self.grasping_state = "lifting"
                    self.grasp_timer = 0
                    print(f"(4)完成grasping,开始lifting")
                
            elif self.grasping_state == "lifting":
                actions[:, 7] = 0.0 # 线速度
                actions[:, 8] = 0.
                # 保持夹爪闭合
                actions[:, 6] = -1
                actions[:, 2] = 2.0
                #if path_tracker.grasping_state == "lifting":
                    #print("lifting:grasp_timer=",path_tracker.grasp_timer)
                
                if self.grasp_timer >= self.lift_time:
                    #self.grasping_state = "completed"
                    #actions[:, 6] = -1
                    cube_height = env._cube_root_states[0, 2].cpu().item()  # 物体高度
                    table_height = env.table_heights[0].cpu().item()        # 桌面高度
                    lift_threshold = 0.05                                   # 物体需高于桌面的阈值（5厘米）
                    cube_pos = env._cube_root_states[0, :3].cpu()
                    ee_pos = env.ee_pos[0].cpu() 
                    obj_ee_distance = torch.norm(cube_pos - ee_pos).item()  # 物体到末端的欧氏距离
                    distance_threshold = 0.08 
                    # 如果物体高于桌面一定高度，且位于夹爪附近，则视为成功
                    if (cube_height - table_height > lift_threshold) and (obj_ee_distance < distance_threshold):# and self.flag==0:
                        print("completed!")
                        print("抓取成功！")
                        #successful_episodes += 1
                        #self.flag=1
                        #total_episodes += 1
                        self.stats.successful += 1
                        self.stats.total += 1
                        env.reset_buf[:]=1
                    else:# self.flag==0:
                        print("completed!")
                        print(f"抓取失败")
                        #total_episodes += 1
                        #self.flag=1
                        self.stats.total += 1
                        env.reset_buf[:]=1
                    #else:
                        #pass
            return actions
        
        def update_arm_position_via_ik(self, target_position, target_rpy):
            device = self.env.device

            # --- 位置控制 ---
            delta_position = self._compute_pid_position(
                current=self.env.ee_pos[0],
                target_position=target_position,
                dt=self.env.dt,
                device=device
            )

            # --- 姿态控制 ---
            delta_orientation = self._compute_pid_orientation(
                current_rpy=self.env.ee_orn[0],
                target_rpy=target_rpy,
                dt=self.env.dt,
                device=device
            )

            # --- 合并控制量 ---
            action_deltas = torch.cat([delta_position, delta_orientation])
            return action_deltas.cpu()

        def _compute_pid_position(self, current, target_position, dt, device):
            current = current.to(device)
            target_position = target_position.to(device)

            error = target_position - current
            derivative = (error - self.prev_position_error) / dt
            self.prev_position_error = error

            self.position_error_integral += error * dt
            self.position_error_integral = torch.clamp(
                self.position_error_integral, -self.max_position_integral, self.max_position_integral
            )

            delta = (
                self.position_kp * error +
                self.position_ki * self.position_error_integral +
                self.position_kd * derivative
            )
            return torch.clamp(delta, -0.07, 0.07)

        def _compute_pid_orientation(self, current_rpy, target_rpy, dt, device):
            current_rpy = euler_from_quat(current_rpy.to(device).unsqueeze(0) )[2]
            target_rpy = euler_from_quat(target_rpy.to(device).unsqueeze(0) )[2]
            
            orientation_error = target_rpy - current_rpy
            derivative = (orientation_error - self.prev_orientation_error) / dt
            self.prev_orientation_error = orientation_error

            self.orientation_error_integral += orientation_error * dt
            self.orientation_error_integral = torch.clamp(
                self.orientation_error_integral, -self.max_orientation_integral, self.max_orientation_integral
            )

            delta = (
                self.orientation_kp * orientation_error +
                self.orientation_ki * self.orientation_error_integral +
                self.orientation_kd * derivative
            )
            return torch.clamp(delta, -0.05, 0.05)

    # 创建环境
    env = create_env(cfg, args)
    tl = 0.1  # 网格大小(米)
    width = 10  # 网格宽度
    height = 10  # 网格高度
    grid_map = GridMap(width=width, height=height, resolution=tl)
    grid_map.add_obstacle(0.0, 0.0, 0.6 ,0.6) 
    grid_map.inflate(robot_radius=0.4)
    obs = env.reset()

    # 重新规划路径并创建新的跟踪器
    path_tracker = plan_new_path(env, grid_map, stats)
    #total_episodes = 1
    #successful_episodes = 0
    # 初始化回合统计变量
    #path_tracker.total_episodes = 1
    #path_tracker.successful_episodes = 0
    #total_episodes += 1
    #开始一个回合
    for i in range(env.max_episode_length):
        if path_tracker.grasping_state == "navigating" or path_tracker.grasping_state == "aligning":
            # 获取当前机器人位置和朝向
            robot_state = env._robot_root_states  
            current_pos = (
                robot_state[0, 0].cpu().item(), 
                robot_state[0, 1].cpu().item()
            )  # 转换为Python标量
            current_heading = euler_from_quat(robot_state[0, 3:7].unsqueeze(0) )[2]
            
            # 获取控制指令
            linear_vel, angular_vel = path_tracker.get_action(current_pos, current_heading)
           
            # 创建动作
            device = env.rl_device
            actions = torch.zeros(env.num_envs, env.num_actions, device=device, dtype=torch.float32)
    
            # 设置机器狗速度命令
            actions[:, 7] = linear_vel 
            actions[:, 8] = angular_vel
            
        elif path_tracker.grasping_state == "approaching" or path_tracker.grasping_state == "grasping" or path_tracker.grasping_state == "lifting":
            actions = torch.zeros(env.num_envs, env.num_actions, device=device, dtype=torch.float32)
            actions = path_tracker.get_grasping_actions(actions)
            if path_tracker.grasping_state == "approaching":
                # 获取物体和末端执行器位置
                cube_pos = env._cube_root_states[0, :3].cpu()
                cube_pos[1]+=0.025
                cube_pos[2]+=0.02
                #cube_orn= env._cube_root_states[0, 3:7].cpu()
                roll = 0.   # 90 度，手指朝左右打开
                pitch = math.pi / 2
                yaw = 0.0

                # 构造四元数
                cube_orn = quat_from_euler_xyz(
                    torch.tensor([roll], device=env.device),
                    torch.tensor([pitch], device=env.device),
                    torch.tensor([yaw], device=env.device)
                ).squeeze(0)
                #cube_orn = quat_from_euler_xyz(torch.tensor([[roll, pitch, yaw]], device=env.device)).squeeze(0)
                #print(f"cube_pos={cube_pos}, cube_orn={cube_orn}")
                pose_deltas = path_tracker.update_arm_position_via_ik(cube_pos,cube_orn)
                actions[:, :6] = pose_deltas
                actions[:, 6] = 1
            if done.any():
                #path_tracker.total_episodes += 1
                print(f"回合重置")
                #path_tracker.flag=0
                obs = env.reset()
                path_tracker = plan_new_path(env, grid_map, stats)  
                continue
        
        # 执行动作
        _, _, done, _, _ = env.step(actions)
        
        
        # 如果有可视化，渲染
        if not args.headless:
            env.render()
        
        # 如果回合结束，重置环境
        '''
        if done.any():
            path_tracker.total_episodes += 1
            print(f"回合重置")
            path_tracker.flag=0
            obs = env.reset()
            path_tracker = plan_new_path(env, grid_map)
        '''
     # 计算并显示最终的成功率
    #success_rate = (successful_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    success_rate = (stats.successful / stats.total) * 100 if stats.total > 0 else 0
    print("=" * 50)
    print(f"仿真结束")
    #print(f"总回合数: {total_episodes}, 成功回合数: {successful_episodes}")
    print(f"总回合数: {stats.total}, 成功回合数: {stats.successful}")
    print(f"成功率: {success_rate:.2f}%")
    print("=" * 50)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # 命令行参数设置
    parser.add_argument("--task", type=str, default="B1Z1ManipLoco", help="Task name")
    parser.add_argument("--headless", action="store_true", default=False, help="Run without rendering")
    parser.add_argument("--horovod", action="store_true", default=False, help="Use horovod for multi-gpu training")
    parser.add_argument("--rl_device", type=str, default="cuda:0", help="Device for RL algorithm")
    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Device for physics simulation")
    parser.add_argument("--graphics_device_id", type=int, default=0, help="Device for rendering")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum number of iterations")
    
    # 额外的参数
    parser.add_argument("--debugvis", action="store_true", default=False, help="Enable debug visualization")
    parser.add_argument("--small_value_set_zero", action="store_true", default=False)
    parser.add_argument("--last_commands", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--control_freq", type=int, default=None)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use_tanh", action="store_true", default=False)
    parser.add_argument("--near_goal_stop", action="store_true", default=False)
    parser.add_argument("--obj_move_prob", type=float, default=0.0)
    parser.add_argument("--roboinfo", action="store_true", default=True, help="Use robot info")
    parser.add_argument("--observe_gait_commands", action="store_true", default=True, help="Observe gait commands")
    parser.add_argument("--no_feature", action="store_true", default=False)
    parser.add_argument("--mask_arm", action="store_true", default=False)
    parser.add_argument("--pitch_control", action="store_true", default=False)
    parser.add_argument("--rand_control", action="store_true", default=False)
    parser.add_argument("--arm_delay", action="store_true", default=False)
    parser.add_argument("--rand_cmd_scale", action="store_true", default=False)
    parser.add_argument("--rand_depth_clip", action="store_true", default=False)
    parser.add_argument("--stop_pick", action="store_true", default=False)
    parser.add_argument("--table_height", type=float, default=None)
    
    args = parser.parse_args()
    
    # 设置评估模式
    is_eval = args.eval
    
    main()