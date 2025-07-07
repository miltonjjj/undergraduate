# 本科毕业设计

本设计基于 [visual_wholebody](https://github.com/Ericonaldo/visual_wholebody/tree/main) 进行修改，使用前需要先编译原始仓库。

## wo_vision_grasp
由路径规划和逆运动学完成的无视觉信息的高层策略

请将以下文件放置在指定路径中：

- 将 `b1z1_base.py`、`b1z1_pickmulti.py`、`b1z1_maniploco.py` 放入：
high-level/envs
- 将 `b1z1_maniploco.yaml` 放入：
high-level/data/cfg
- 将 `path_planner.py`、`play_maniploco.py` 放入：
high-level

## go2arx5
将物理模型由b1z1替换为go2arx5

将对应的文件夹替换为项目中的 `go2arx5` 文件夹即可。

