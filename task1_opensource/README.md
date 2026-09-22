# Task 1 最小开源复现包

本目录是 AgiBot G1 任务 1 的最小可复现版本，只保留当前使用的方案：

`g1_button.json` 场景 → 头部 RGB-D → RTAB-Map 建图/定位 → Nav2 Hybrid-A* + MPPI → A→B→C。

历史尝试、诊断脚本、日志和缓存均未包含。OrcaLab 本体及机器人/场景资产不在本包内；场景 JSON 已包含，但其 `assets/...` 引用需要在目标机器的 OrcaLab 资产库中存在。常用 RTAB-Map 地图已随包提供。

## 目录

```text
task1_opensource/
├── g1_button.json                 # 需要在 OrcaLab 中打开并 Play 的场景
├── config/nav2_truth.yaml         # 当前 Nav2 参数
├── src/                           # 当前方案及其运行时依赖
├── run_task1.sh                   # 定位 + A→B→C 一键入口
├── tests/                         # 轻量合同测试
├── data/world_anchored_rtabmap_20260821T195241+0800/
│   ├── rtabmap.db                 # 已建好的常用 RTAB-Map 数据库
│   ├── mapping_manifest.json
│   ├── world_to_map_calibration.json
│   └── map/                       # 二维占据栅格
└── data/validation_<timestamp>/   # 运行结果
```

`src/` 只保留导航运行时会加载的模块；相机配置脚本位于根目录，因为主入口会在场景 reset 后直接调用它。建图航点记录和独立建图辅助脚本已移除。`run_world_anchored_rtabmap_mapping.py` 仅保留导航启动时复用的场景读取、相机和清理函数，不作为开源包的建图入口。

## 环境准备

已验证的基线：Ubuntu 24.04、ROS 2 Jazzy、OrcaLab、Python 3.12 的 `orcalab` 环境。

1. 安装并启动 OrcaLab，确认编辑服务 `127.0.0.1:50151`、仿真服务 `127.0.0.1:50051` 可访问。
2. 在 OrcaLab 中打开本目录的 `g1_button.json`，点击 **Play**。不要同时运行其他会步进 OrcaGym 的控制程序。
3. 准备 ROS 2 与 Python 环境：

   ```bash
   source /opt/ros/jazzy/setup.bash
   export ORCALAB_PYTHON=/home/dan/miniconda3/envs/orcalab/bin/python
   export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"
   ```

   Python 环境需要 `orca_gym`、`numpy`、`Pillow`、`grpcio`、`mujoco`；ROS 需要 `rclpy`、`tf2_ros`、`sensor_msgs`、`nav_msgs`、`nav2_msgs`、`rtabmap_ros` 和 Nav2 MPPI/Smac 插件。

## 内置常用地图

包内已包含 `data/world_anchored_rtabmap_20260821T195241+0800/`，包括 RTAB-Map 数据库、二维占据栅格、建图 manifest 和 world→map 标定。正常复现不需要重新建图，直接运行：

```bash
cd task1_opensource
./run_task1.sh
```

默认验证速度为 `0.8 m/s`，也可以指定速度：

```bash
./run_task1.sh 0.8
```

## 复现 A→B→C

默认使用包内常用地图；如替换为其他兼容地图，可将建图会话目录作为第一个参数：

```bash
./run_task1.sh data/world_anchored_rtabmap_20260920T120000 0.8
```

结果写入 `data/validation_<timestamp>/`。如需调试，可在命令末尾追加入口脚本支持的参数；自适应路线画像是对照选项：

```bash
./run_task1.sh data/world_anchored_rtabmap_20260920T120000 0.8 \
  --validation-adaptive-route-profile
```

运行前须确认场景已 Play，且当前没有残留的 ROS/OrcaGym 控制进程。正常结束后重点查看 `localization_acceptance.json` 中的 `passed`、RGB-D 失败组、定位丢失率和碰撞审计。

## 测试

不连接 OrcaLab 也可以运行纯合同测试：

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src python -m pytest -q tests/test_task1_scene_contract.py
```

## 说明

- 本包不提供 OrcaLab 商业/内部资产；`g1_button.json` 只是场景布局和资产引用。
- 内置常用地图数据库约 343 MB；运行产生的验证结果保存在 `data/validation_<timestamp>/`。
- 当前方案不使用 wheel odom、不连续发布 MuJoCo truth odom；MuJoCo 世界位姿只用于视觉里程计初始化和离线审计。
