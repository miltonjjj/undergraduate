import torch
import numpy as np
from astar import find_path
from typing import Tuple, List, Optional, Iterable
import heapq

class GridMap:
    
    def __init__(self, width: float, height: float, resolution: float = 0.1):
        self.resolution = resolution
        self.grid_width = int(width / resolution)  # 计算网格的列数
        self.grid_height = int(height / resolution)  # 计算网格的行数
        # 创建一个空的网格地图，0表示可通行，1表示障碍物
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
   
    def inflate(self, robot_radius:float):
        inflate_radius = int(np.ceil(robot_radius / self.resolution))
        inflated_grid = np.copy(self.grid)  # 不要在遍历时修改原始 grid

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                if self.grid[i, j] == 1:
                    # 对障碍物周围一定范围内进行膨胀
                    for dx in range(-inflate_radius, inflate_radius + 1):
                        for dy in range(-inflate_radius, inflate_radius + 1):
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < self.grid_width and 0 <= nj < self.grid_height:
                                # 不覆盖已有障碍
                                if inflated_grid[ni, nj] == 0:
                                    inflated_grid[ni, nj] = 2

        self.grid = inflated_grid  # 更新地图


    def add_obstacle(self, x: float, y: float, width: float, height: float):
        center_x, center_y = self.world_to_grid(x, y)
        half_width = int(np.round((width / 2) / self.resolution))
        half_height = int(np.round((height / 2) / self.resolution))

        for i in range(center_x - half_width, center_x + half_width):
            for j in range(center_y - half_height, center_y + half_height):
                if 0 <= i < self.grid_width and 0 <= j < self.grid_height:
                    self.grid[i, j] = 1

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        # 通过坐标转换公式将世界坐标转换为网格坐标
        grid_x = int((x + self.grid_width * self.resolution / 2) / self.resolution)
        grid_y = int((y + self.grid_height * self.resolution / 2) / self.resolution)
        # 返回转换后的网格坐标，确保不会超出地图边界
        return max(0, min(grid_x, self.grid_width - 1)), max(0, min(grid_y, self.grid_height - 1))
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        # 通过坐标转换公式将网格坐标转换回世界坐标
        x = grid_x * self.resolution - self.grid_width * self.resolution / 2
        y = grid_y * self.resolution - self.grid_height * self.resolution / 2
        return x, y
    
    def is_valid_position(self, grid_x: int, grid_y: int) -> bool:
        """检查位置是否有效且可通行"""
        # 判断该位置是否在网格内且不被障碍物占据
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.grid[grid_x, grid_y] == 0  # 如果网格是0，则表示可通行
        return False
    def clear_obstacles(self):
        """清除所有障碍物"""
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int8)

class PathPlanner:
    """路径规划器,使用自定义A*算法实现路径规划"""
    
    def __init__(self, grid_map: GridMap):
        """
        初始化路径规划器
        
        Args:
            grid_map: 网格地图
        """
        self.grid_map = grid_map  # 将网格地图传入路径规划器
        
    def plan_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        规划从起始位置到目标位置的路径
        
        Args:
            start_pos: 起始位置的世界坐标 (x, y)
            goal_pos: 目标位置的世界坐标 (x, y)
            
        Returns:
            路径点列表，每个点为世界坐标 (x, y)
        """
        # 将世界坐标转换为网格坐标
        start_grid = self.grid_map.world_to_grid(*start_pos)
        goal_grid = self.grid_map.world_to_grid(*goal_pos)
        
        # 使用自定义A*算法计算路径
        path_grid = self.a_star_search(start_grid, goal_grid)
        
        # 如果没有路径，返回空列表
        if path_grid is None:
            return []
        
        # 将网格坐标转换为世界坐标，并返回路径
        path_world = [self.grid_map.grid_to_world(x, y) for x, y in path_grid]
        return path_world
    
    def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        使用A*算法寻找从起点到目标点的路径
        
        Args:
            start: 起始网格坐标 (x, y)
            goal: 目标网格坐标 (x, y)
            
        Returns:
            找到的路径,如果没有路径则返回None
        """
        # 定义启发式函数 - 使用曼哈顿距离
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # 开放列表和关闭列表
        open_list = []
        closed_set = set()
        
        # 使用字典记录每个节点的g值和父节点
        g_scores = {start: 0}  # 从起点到该节点的实际代价
        came_from = {}  # 记录路径
        
        # 将起始节点添加到开放列表
        # 格式: (f_score, 计数器, node)，使用f_score作为优先级，计数器用于打破平局
        counter = 0
        heapq.heappush(open_list, (heuristic(start, goal), counter, start))
        
        while open_list:
            # 获取当前f值最小的节点
            _, _, current = heapq.heappop(open_list)
            
            # 如果到达目标，重建路径并返回
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # 反转路径，使其从起点到终点
            
            # 将当前节点加入关闭列表
            closed_set.add(current)
            
            # 检查所有邻居
            x, y = current
            # 8个方向的邻居
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                         (1, 1), (-1, -1), (1, -1), (-1, 1)]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                # 检查邻居是否有效且可通行
                if not self.grid_map.is_valid_position(nx, ny):
                    continue
                
                # 如果邻居在关闭列表中，跳过
                if neighbor in closed_set:
                    continue
                
                # 计算从起点到邻居的实际代价
                # 对角线移动的代价为sqrt(2)，直线移动的代价为1
                if abs(dx) + abs(dy) == 2:  # 对角线移动
                    tentative_g_score = g_scores[current] + 1.414  # sqrt(2)
                else:  # 直线移动
                    tentative_g_score = g_scores[current] + 1
                
                # 如果邻居不在g_scores中或新的g值更小
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    # 更新数据
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    
                    # 将邻居添加到开放列表
                    counter += 1
                    heapq.heappush(open_list, (f_score, counter, neighbor))
        
        # 如果开放列表为空但仍未到达目标，则无路径
        return None
    '''
    def path_to_actions(self, path: List[Tuple[float, float]], dt: float) -> List[Tuple[float, float]]:
        """
        将路径转换为机器人动作指令
        
        Args:
            path: 路径点列表
            dt: 控制周期（时间间隔）
            
        Returns:
            动作列表，每个动作为 (vx, vy),表示机器人在x和y方向上的速度
        """
        if not path or len(path) < 2:
            return []  # 如果路径为空或只有一个点，则无法规划动作
        
        actions = []
        for i in range(len(path) - 1):
            x1, y1 = path[i]  # 当前点
            x2, y2 = path[i + 1]  # 下一个点
            
            # 计算从当前点到下一个点的方向向量
            dx, dy = x2 - x1, y2 - y1
            distance = np.sqrt(dx**2 + dy**2)  # 计算两点之间的距离
            
            # 归一化方向向量，得到单位方向
            if distance > 0:
                dx, dy = dx / distance, dy / distance
            
            # 设置速度为固定值，通常可以根据需要调整
            speed = 0.5  # 机器人移动的速度
            vx, vy = dx * speed, dy * speed  # 计算每个时刻的速度
            
            actions.append((vx, vy))  # 将速度指令加入到动作列表
        
        return actions
        '''