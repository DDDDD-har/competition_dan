# competition_dan

competition_dan 任务复现代码和配置。为避免在 GitHub 上直接展开源码，两个独立入口以加密 7z 包发布：

- `task1_opensource.7z`：解压为 `task1_opensource/`，用于 G1 导航、RTAB-Map 定位和 Nav2 A→B→C。
- `task2.7z`：解压为 `task2/`，用于 G1 OmniPicker 按钮任务和 openpi v10 checkpoint 推理。

两项任务都依赖 OrcaLab 仿真环境；任务 2 还需要单独运行 openpi 推理服务。

## 复现前提

已验证的基础环境是 Ubuntu 24.04、Python 3.12 和 OrcaLab。两个加密代码包直接存放在 GitHub 仓库根目录：新克隆会一并下载；已有克隆先更新到最新 `main`。地图包和权重包则需另外从 Hugging Face 下载。安装 7-Zip 后执行：

```bash
sudo apt update
sudo apt install -y p7zip-full
git clone https://github.com/DDDDD-har/competition_dan.git
cd competition_dan
test -f task1_opensource.7z
test -f task2.7z
7z x -aoa -p'B8vp3jk4k0EVNFZWMQ7u' task1_opensource.7z
7z x -aoa -p'B8vp3jk4k0EVNFZWMQ7u' task2.7z
test -f task1_opensource/run_task1.sh
test -f task2/run_v10_9999_local_noseek.sh
```

已有仓库克隆若缺少上述归档，先运行 `git pull --ff-only origin main` 再解压。两个包都开启了 AES 和文件头加密。解压后，具体 ROS/OpenPI 依赖可查看 `task1_opensource/README.md` 和 `task2/README.md`；外部资产的下载与解压路径以本 README 为准。两个项目包不含任务 1 的地图数据和任务 2 的 checkpoint，这两项资产需要继续按下一节从 Hugging Face 下载。地图包解压后会提供任务 1 运行所需的 RTAB-Map 数据库、栅格图、manifest 和标定文件。

不要在同一个 OrcaLab 实例中同时运行任务 1 和任务 2 的控制脚本，也不要启动多个会步进 OrcaGym 的进程。

## 下载大文件

GitHub 不存储超过单文件限制的训练 checkpoint 和 RTAB-Map 数据库。这两个运行时资产托管在公开 Hugging Face 数据集：

[`dan5433/competition_dan`](https://huggingface.co/datasets/dan5433/competition_dan)

安装 Hugging Face CLI 后，在项目根目录只下载地图和权重包（公开数据集通常无需登录）：

```bash
hf download dan5433/competition_dan \
  'world_anchored_rtabmap_20260821T195241+0800.7z' \
  9999.7z \
  --repo-type dataset \
  --local-dir .
```

也可以不安装 CLI，用 `curl` 直接下载：

```bash
HF_ASSETS_URL="https://huggingface.co/datasets/dan5433/competition_dan/resolve/main"
curl -fL "$HF_ASSETS_URL/world_anchored_rtabmap_20260821T195241%2B0800.7z" \
  -o 'world_anchored_rtabmap_20260821T195241+0800.7z'
curl -fL -O "$HF_ASSETS_URL/9999.7z"
```

单文件链接：

- [RTAB-Map 建图会话](https://huggingface.co/datasets/dan5433/competition_dan/resolve/main/world_anchored_rtabmap_20260821T195241%2B0800.7z)（含 `rtabmap.db`）
- [任务 2 checkpoint `9999.7z`](https://huggingface.co/datasets/dan5433/competition_dan/resolve/main/9999.7z)（v10 第 9999 步）

四个 7z 包（两个项目包、地图包和权重包）的密码相同：`B8vp3jk4k0EVNFZWMQ7u`。地图包解压到任务 1 的 `data/` 目录，权重包直接解压到项目根目录；归档自带 `9999/` 顶层目录：

```bash
7z x -aoa -p'B8vp3jk4k0EVNFZWMQ7u' \
  world_anchored_rtabmap_20260821T195241+0800.7z \
  -o"$PWD/task1_opensource/data"
7z x -aoa -p'B8vp3jk4k0EVNFZWMQ7u' 9999.7z -o"$PWD"

test -f task1_opensource/data/world_anchored_rtabmap_20260821T195241+0800/rtabmap.db
test -d 9999/params
test -f 9999/assets/g1_button_scripted_v10/norm_stats.json
```

权重包 `9999.7z` 是 v10 第 9999 步 checkpoint 的加密包，解压后目录名为 `9999/`。上面三条 `test` 均无输出且返回 0，说明目录层级正确。

## 任务 1：导航复现

1. 安装并启动 OrcaLab，确认编辑服务 `127.0.0.1:50151` 和仿真服务 `127.0.0.1:50051` 可访问。
2. 在 OrcaLab 中打开 `task1_opensource/move.json` 并点击 **Play**。
3. 准备 ROS 2 Jazzy 和 Python 环境：

   ```bash
   source /opt/ros/jazzy/setup.bash
   export ORCALAB_PYTHON=/path/to/orcalab/bin/python
   export PYTHONPATH="$PWD/task1_opensource:$PWD/task1_opensource/src:${PYTHONPATH:-}"
   ```

   Python 环境需要 `orca_gym`、`numpy`、`Pillow`、`grpcio`、`mujoco`；ROS 需要 `rclpy`、`tf2_ros`、`sensor_msgs`、`nav_msgs`、`nav2_msgs`、`rtabmap_ros` 和 Nav2 MPPI/Smac 插件。
4. 确认地图包已经解压到 `task1_opensource/data/world_anchored_rtabmap_20260821T195241+0800/`，然后运行 A→B→C：

   ```bash
   cd task1_opensource
   ./run_task1.sh
   ```

   默认速度为 `0.8 m/s`。该入口在导航前会 reset 场景，并清空评分服务器地址、关闭评分录像；按本节命令运行时只使用本机 OrcaLab，不连接中央评分服务。结果写入 `task1_opensource/data/validation_<timestamp>/`，重点查看 `localization_acceptance.json` 和 `global_geometry_acceptance.json`。

   新的 world→map 标定只会在定位验收和多点全局几何验收都通过后写回地图目录。若路线或几何审计未通过，本次运行目录仍会保留诊断结果，但不会用未通过的拟合结果替换已接受标定。

更完整的地图、参数和故障排查说明见解压后的 `task1_opensource/README.md`。

## 任务 2：按钮策略复现

任务 2 不是独立可运行的 Python 包。它需要：

- OrcaLab 中打开并 Play `task2/g1_button.json`；
- NVIDIA GPU（策略服务实测约占 22 GB 显存，建议至少 24 GB；与 OrcaLab 共用一张卡时建议至少 32 GB）；
- openpi 上游源码和 `task2/openpi_task2.patch`；
- `task2/` 内随附的 SouthGrid 任务运行时模块；
- Hugging Face 上的 `9999.7z` 解压出的 checkpoint。

具体步骤：

1. 按解压后的 `task2/README.md` 检出已验证的 openpi 提交 `981483d`，应用随包提供的补丁，并确认配置存在：

   ```bash
   export SOUTHGRID_ROOT=/path/to/competition_dan
   git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
   cd openpi
   git checkout 981483d
   git submodule update --init --recursive
   git apply "$SOUTHGRID_ROOT/task2/openpi_task2.patch"
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   export OPENPI_ROOT="$PWD"

   test -f "$OPENPI_ROOT/src/openpi/policies/g1_omnipicker_policy.py"
   grep -n "pi05_g1_scripted_lora_v10" "$OPENPI_ROOT/src/openpi/training/config.py"
   ```

   上游 `Physical-Intelligence/openpi` 不含这两项定制内容，不能直接用于本 checkpoint；`task2.7z` 已包含所需补丁。

2. 如果已按「下载大文件」解压权重，使 `CKPT_DIR` 指向包含 `params/` 的目录：

   ```bash
   export SOUTHGRID_ROOT=/path/to/competition_dan
   export CKPT_DIR="$SOUTHGRID_ROOT/9999"
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

4. 在 OrcaLab 打开 `task2/g1_button.json` 并点击 **Play**。另开终端设置推理客户端 Python 环境：

   ```bash
   export PYTHON=/path/to/orcalab/bin/python
   export OPENPI_ROOT=/path/to/openpi
   # 如 orca_gym 不在当前环境中：
   # export ORCA_GYM_ROOT=/path/to/OrcaGym
   ```

   其中 Python 环境需要 `orca_gym`、`openpi_client`、`numpy`、`cv2`、`pyyaml`、`scipy`、`h5py` 和 `av`，且 OrcaLab gRPC 服务应在 `localhost:50051`。

5. 启动一轮四色推理：

   ```bash
   cd /path/to/competition_dan/task2
   PYTHON="$PYTHON" bash run_v10_9999_local_noseek.sh
   ```

当前推理客户端使用 `exec_horizon=50`、`action_repeat=10`、`max_steps=1000`，初始保持 10 步并锁定左臂。

`run_v10_9999_local_noseek.sh`、日志目录中的 `v10_9999` 以及脚本中的 v10 字样都对应当前发布的 v10 策略和第 9999 步 checkpoint。

任务 2 的相机映射和参数限制见解压后的 `task2/README.md`。

## 复现边界

- 仓库不包含 OrcaLab 的商业/内部机器人和场景资产；Task1 的 `move.json` 与 Task2 的 `g1_button.json` 只保存场景布局及资产引用。
- GitHub 只提供两个加密项目包，不提供展开的源码目录。
- 任务 1 的 RTAB-Map 数据库和任务 2 的 checkpoint 必须另外从 Hugging Face 下载并解压；数据库随地图包解压到任务 1 默认目录，权重随 checkpoint 包解压到项目根目录的 `9999/`。
- 任务 2 的 checkpoint 依赖项目定制 openpi；只有上游 openpi 仓库时无法启动该策略。
- 任务 2 所需的 SouthGrid 任务运行时模块已复制到 `task2/`；`orca_gym` 等第三方运行库仍需安装。
- 推理日志、验证会话和本地缓存不会提交到 GitHub。
