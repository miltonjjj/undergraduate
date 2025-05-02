import numpy as np
import os
import torch
import cv2
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict
import wandb

from .b1z1_pickmulti import B1Z1PickMulti
from .b1z1_base import B1Z1Base

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
import torchvision.transforms as transforms
import math

class B1Z1ManipLoco(B1Z1PickMulti):
    def __init__(self,table_height=None, *args, **kwargs):
        super().__init__(table_height=0.6,*args, **kwargs)
        #self.near_goal_stop = self.cfg["env"].get("near_goal_stop", False)
        #self.obj_move_prob = self.cfg["env"].get("obj_move_prob", 0.0)
        #self.floating_base = self.cfg["env"].get("floatingBase", False)
        #table_height=0.6
        #self.table_heights_fix = table_height
        #self.table_dims.z=0.6
    
    def _reset_envs(self,env_ids):
        B1Z1Base._reset_envs(self,env_ids)
        #if env_ids is None:
           # env_ids = torch.arange(self.num_envs, device=self.device)
        num_group= self.num_envs //1
        square_box_indices_np = np.array([[i]for i in range(num_group)]).reshape(1,-1).squeeze()
        square_box_indices = torch.from_numpy(square_box_indices_np).to(self.device)
            
        squarebox_success_time = self.success_counter[square_box_indices].sum().item(), self.episode_counter[square_box_indices].sum().item()
           
        squarebox_success_rate = min(squarebox_success_time[0], squarebox_success_time[1]) / max(squarebox_success_time[1], 1)
        
    def _reset_objs(self, env_ids):
        if len(env_ids)==0:
            return
        
        # self._cube_root_states[env_ids] = self._initial_cube_root_states[env_ids]
        self._cube_root_states[env_ids, 0] = 0.0
        self._cube_root_states[env_ids, 0] += torch_rand_float(-0.15, 0., (len(env_ids), 1), device=self.device).squeeze(1)
        self._cube_root_states[env_ids, 1] = 0.0
        self._cube_root_states[env_ids, 1] += torch_rand_float(0., 0.1, (len(env_ids), 1), device=self.device).squeeze(1)
        
        self._cube_root_states[env_ids, 2] = self.table_heights[env_ids] + self.init_height[env_ids]
        #rand_yaw_box = torch_rand_float(-3.15, 3.15, (len(env_ids), 1), device=self.device).squeeze(1)
        rand_yaw_box = torch_rand_float(0.0, 0.0, (len(env_ids), 1), device=self.device).squeeze(1)
        
        if True: # self.global_step_counter < 25000:
            self._cube_root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(0*rand_yaw_box, 0*rand_yaw_box, rand_yaw_box), self.init_quat[env_ids]) # Make sure to learn basic grasp
        else:
            rand_r_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            rand_p_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            self._cube_root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(rand_r_box, rand_p_box, rand_yaw_box), self.init_quat[env_ids])
        self._cube_root_states[env_ids, 7:13] = 0.
    
    

    