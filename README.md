# southgrid

Southgrid 任务复现代码和配置。仓库分成两个独立入口：

- [`task1_opensource/`](task1_opensource/README.md)：G1 导航任务，RTAB-Map 定位和 Nav2 A→B→C。
- [`task2/`](task2/README.md)：G1 OmniPicker 按钮任务，使用 openpi v10 checkpoint 做纯推理。

两项任务都依赖 OrcaLab 仿真环境；任务 2 还需要单独运行 openpi 推理服务。

## 复现前提

已验证的基础环境是 Ubuntu 24.04、Python 3.12 和 OrcaLab。具体 ROS/OpenPI 依赖见对应目录的 README。先克隆仓库并进入项目根目录：

```bash
git clone https://github.com/DDDDD-har/southgrid.git
cd southgrid
```

不要在同一个 OrcaLab 实例中同时运行任务 1 和任务 2 的控制脚本，也不要启动多个会步进 OrcaGym 的进程。

## 下载大文件

GitHub 不存储超过单文件限制的训练 checkpoint 和 RTAB-Map 数据库。大文件托管在公开 Hugging Face 数据集：

[`dan5433/southgrid-assets`](https://huggingface.co/datasets/dan5433/southgrid-assets)

安装并登录 Hugging Face CLI 后，在项目根目录执行：

```bash
hf download dan5433/southgrid-assets \
  --repo-type dataset \
  --local-dir .
```

这会按仓库路径恢复文件：

```text
task1_opensource/data/world_anchored_rtabmap_20260821T195241+0800/rtabmap.db
task2/9999.zip
```

也可以单独下载：

- [RTAB-Map 数据库](https://huggingface.co/datasets/dan5433/southgrid-assets/resolve/main/task1_opensource/data/world_anchored_rtabmap_20260821T195241%2B0800/rtabmap.db)
- [任务 2 checkpoint `9999.zip`](https://huggingface.co/datasets/dan5433/southgrid-assets/resolve/main/task2/9999.zip)

`9999.zip` 约 9.4 GB，下载完成后不要把 zip 直接传给推理服务；任务 2 需要先解压。若该链接暂时返回 404，说明 Hugging Face 的大文件提交尚未完成，请先在数据集页面确认文件已经出现。

## 任务 1：导航复现

1. 安装并启动 OrcaLab，确认编辑服务 `127.0.0.1:50151` 和仿真服务 `127.0.0.1:50051` 可访问。
2. 在 OrcaLab 中打开 `task1_opensource/g1_button.json` 并点击 **Play**。
3. 准备 ROS 2 Jazzy 和 Python 环境：

   ```bash
   source /opt/ros/jazzy/setup.bash
   export ORCALAB_PYTHON=/home/dan/miniconda3/envs/orcalab/bin/python
   export PYTHONPATH="$PWD/task1_opensource:$PWD/task1_opensource/src:${PYTHONPATH:-}"
   ```

   Python 环境需要 `orca_gym`、`numpy`、`Pillow`、`grpcio`、`mujoco`；ROS 需要 `rclpy`、`tf2_ros`、`sensor_msgs`、`nav_msgs`、`nav2_msgs`、`rtabmap_ros` 和 Nav2 MPPI/Smac 插件。
4. 运行内置地图上的 A→B→C：

   ```bash
   cd task1_opensource
   ./run_task1.sh
   ```

   默认速度为 `0.8 m/s`。结果写入 `task1_opensource/data/validation_<timestamp>/`，重点查看其中的 `localization_acceptance.json`。不连接 OrcaLab 的静态合同测试可以这样运行：

   ```bash
   source /opt/ros/jazzy/setup.bash
   PYTHONPATH=src python -m pytest -q tests/test_task1_scene_contract.py
   ```

更完整的地图、参数和故障排查说明见 [`task1_opensource/README.md`](task1_opensource/README.md)。

## 任务 2：按钮策略复现

任务 2 不是独立可运行的 Python 包。它需要：

- OrcaLab 中打开并 Play `task2/g1_button.json`；
- 含 `pi05_g1_scripted_lora_v10` 配置的 openpi 源码；
- 外部 SouthGrid 源码树，用于提供 `conf`、`controllers`、`dataStorage`、`envs` 等模块；
- `task2/9999.zip` 解压出的 checkpoint。

具体步骤：

1. 按 [`task2/README.md`](task2/README.md) 安装 openpi，并确认配置存在：

   ```bash
   grep -n "pi05_g1_scripted_lora_v10" "$OPENPI_ROOT/src/openpi/training/config.py"
   ```

2. 解压 checkpoint，使 `CKPT_DIR` 指向包含 `params/` 的目录：

   ```bash
   mkdir -p /path/to/checkpoints
   unzip task2/9999.zip -d /path/to/checkpoints
   export CKPT_DIR=/path/to/checkpoints/9999
   test -d "$CKPT_DIR/params"
   test -f "$CKPT_DIR/assets/g1_button_scripted_v10/norm_stats.json"
   ```

3. 在 openpi 仓库终端启动策略服务：

   ```bash
   cd "$OPENPI_ROOT"
   uv run scripts/serve_policy.py --port 8010 policy:checkpoint \
     --policy.config=pi05_g1_scripted_lora_v10 \
     --policy.dir="$CKPT_DIR"
   ```

   看到 `server listening` 后保持该终端运行。

4. 在 OrcaLab 打开 `task2/g1_button.json` 并点击 **Play**。另开终端设置仿真依赖和 SouthGrid 源码路径：

   ```bash
   export SOUTHGRID_SRC=/path/to/SouthGrid/src
   export PYTHON=/path/to/orcalab/bin/python
   # 如 orca_gym 不在当前环境中：
   # export ORCA_GYM_ROOT=/path/to/OrcaGym
   ```

   其中 Python 环境需要 `orca_gym`、`openpi_client`、`cv2` 和 `pyyaml`，且 OrcaLab gRPC 服务应在 `localhost:50051`。

5. 启动一轮四色评测：

   ```bash
   cd /path/to/southgrid/task2
   SOUTHGRID_SRC="$SOUTHGRID_SRC" PYTHON="$PYTHON" \
     bash run_v10_9999_local_noseek.sh
   ```

   多跑几轮可设置 `EPISODES`：

   ```bash
   EPISODES=3 bash run_v10_9999_local_noseek.sh
   ```

日志位于 `task2/logs/v10_9999_<轮数>x4/eval.log`。每个颜色的 `finished:` 行包含 `touched` 和 `pressed` 结果；脚本使用 `exec_horizon=50`、`action_repeat=10`、`max_steps=2000`，并在触碰后提前结束该颜色。

任务 2 的相机映射和参数限制见 [`task2/README.md`](task2/README.md)。不要添加已移除的 `--score-seek`、`--min-score-span`、`--p2-wait`、`--hold-render-hz` 或 `--score-hold-s` 参数。

## 复现边界

- 仓库不包含 OrcaLab 的商业/内部机器人和场景资产；`g1_button.json` 只保存场景布局及资产引用。
- 任务 1 的 RTAB-Map 数据库和任务 2 的 checkpoint 必须先从 Hugging Face 下载。
- 任务 2 的 openpi 配置和 SouthGrid 运行时源码不是本仓库的一部分，必须按 `task2/README.md` 提供路径。
- 评测日志、验证会话和本地缓存不会提交到 GitHub。
