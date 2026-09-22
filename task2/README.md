# Task2 v10 纯推理

G1 OmniPicker 按按钮任务的 v10 推理。脚本按策略输出执行，不把末端拉到得分点，也不为了凑评分时间停在原地。

2026-09-22 从 SouthGrid 的 `src/examples/inference/g1_omnipicker/eval_g1_omnipicker_lerobot.py` 同步，并将任务所需的 SouthGrid Python 运行时模块复制到本目录。评测入口优先使用 `task2/` 内的副本。

## 目录

| 文件 | 作用 |
|---|---|
| `eval_g1_omnipicker_lerobot.py` | 推理客户端 |
| `run_v10_9999_local_noseek.sh` | 启动一轮或多轮四色评测 |
| `g1_button.json` | 按钮关卡 |
| `9999.zip` | v10 checkpoint `exp01/9999` 的打包 |
| `conf/`、`controllers/`、`dataCollectionManager/`、`dataStorage/`、`devices/`、`scene/`、`task/` | 任务所需的本地 SouthGrid 运行时模块 |
| `logs/` | 本脚本写出的评测日志 |

## 推理服务

推理服务是 [openpi](https://github.com/Physical-Intelligence/openpi) 的 `scripts/serve_policy.py`。它和仿真客户端分开，换一台电脑时按下面三步做。下面的路径都是本机自己选的，没有写死。

需要一块 NVIDIA GPU，推理大约要 8 GB 以上显存，系统用 Ubuntu。

### 1. 下载并安装 openpi

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 之后：

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
export OPENPI_ROOT="$PWD"
```

`GIT_LFS_SKIP_SMUDGE=1` 是为了拉 LeRobot 依赖时跳过大文件。

配置名 `pi05_g1_scripted_lora_v10` 不在上游仓库里，是在这份 openpi 上加进 `src/openpi/training/config.py` 的。新机器上先确认：

```bash
grep -n "pi05_g1_scripted_lora_v10" "$OPENPI_ROOT/src/openpi/training/config.py"
```

搜不到就不要用未改过的上游仓库直接启动。把带这个配置的 openpi 拷过去，或把对应的 `TrainConfig` 加进刚 clone 的仓库。配置里的资源目录如果还写着训练机的绝对路径，可以忽略：服务实际用的是 checkpoint 自带的 `assets/`。

### 2. 解压权重

本目录的 `9999.zip` 就是 v10 的 `exp01/9999`。把它拷到新机器后解压。解压结果里要有 `params/` 和 `assets/g1_button_scripted_v10/norm_stats.json`。不要把 zip 本身传给服务。

```bash
unzip 9999.zip -d /path/to/checkpoints
export CKPT_DIR=/path/to/checkpoints/9999
```

`/path/to/checkpoints` 换成这台机器上的目录。`CKPT_DIR` 必须指到含 `params/` 的那一层，而不是 zip 所在的上一级。

### 3. 启动服务

在 openpi 仓库根目录执行。`--policy.dir` 用上面的 `CKPT_DIR`，不要写训练机上的路径。

```bash
cd "$OPENPI_ROOT"
uv run scripts/serve_policy.py --port 8010 policy:checkpoint \
  --policy.config=pi05_g1_scripted_lora_v10 \
  --policy.dir="$CKPT_DIR"
```

日志出现 `server listening` 后再跑评测。客户端默认连 `localhost:8010`。改了 `--port` 的话，评测脚本里的 `--port` 也要改成同一个。

## 跑评测

仿真侧另有要求：OrcaLab 已打开按钮关卡，gRPC 在 `localhost:50051`；Python 环境里有 `orca_gym`、`openpi_client`、`cv2`、`pyyaml`。任务所需的 SouthGrid Python 模块已随本目录提供，不再要求完整 SouthGrid 源码树。

```bash
export PYTHON=/path/to/env/bin/python
# 若要使用另一份 SouthGrid 源码，可显式覆盖本地副本：
# export SOUTHGRID_SRC=/path/to/SouthGrid/src
# 若当前 Python 环境里没有 orca_gym，再把它的仓库加进来：
# export ORCA_GYM_ROOT=/path/to/OrcaGym
bash run_v10_9999_local_noseek.sh
```

默认情况下脚本会使用 `task2/` 内的模块。`orca_gym`、`openpi_client`、OpenCV、SciPy、h5py、PyAV 等第三方包仍需在 Python 环境中安装；它们不是 SouthGrid 脚本，可以通过 `ORCA_GYM_ROOT` 追加 OrcaGym 源码路径。

在本目录下执行。默认 1 轮、红绿蓝黄各一次。三轮：

```bash
EPISODES=3 bash run_v10_9999_local_noseek.sh
```

日志在 `logs/v10_9999_<轮数>x4/eval.log`。

固定参数：

- `exec_horizon=50`：一次执行模型给出的 50 步
- `action_repeat=10`：每个动作重复 10 个控制步，对齐 20 fps 采集
- `max_steps=2000`
- 碰到按钮（关节位移 ≥ 1 mm）后再保压 40 步结束该颜色
- 本机评分，不上报

要改步数、horizon 或重复次数，直接改脚本里的这几个参数，或自己调用 `eval_g1_omnipicker_lerobot.py`。不要再加 `--score-seek`、`--min-score-span`、`--p2-wait`、`--hold-render-hz`、`--score-hold-s`，这些参数已经从脚本里删掉了。

## 相机

策略要三路图：`cam_head`、`cam_wrist_l`、`cam_wrist_r`。按钮关卡没有单独的腕部传感器，左腕图用的是左相机：

- `camera_head`，7090 → `cam_head`
- `camera_left`，7080 → `cam_wrist_l`
- `camera_right`，7070 → `cam_wrist_r`

这是 v10 采集时的对应关系。不要加 `--enable_wrist_l`，那个开关会去订关卡里不存在的 `camera_wrist_l_color`，并把 7070 和 7080 对调。

## 怎么读结果

日志里每个颜色有一行 `finished:`。

- `touched=True`：按钮位移达到 1 mm，算碰到
- `pressed=True`：按钮位移达到 0.5 mm
- 官方分看末端到按钮 site 的距离，大约 5 cm 就是接近满分。满分不等于按到了按钮

2026-09-22 用这套参数跑过 1 轮四色：官方分 39.8657/39.8657，四色都没碰到。黄钮最大位移 0.17 mm。左相机开着再跑一轮是 39.8659，黄钮 0.26 mm，仍然没碰到。
