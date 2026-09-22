# southgrid

本仓库包含 Southgrid 任务复现代码和配置。`task1_opensource/` 是任务 1 的最小开源复现包，使用方式请参阅其 [README](task1_opensource/README.md)。

## 大文件

GitHub 仓库不直接存储训练检查点压缩包和 RTAB-Map 数据库。它们托管在 Hugging Face 数据集：

[`dan5433/southgrid-assets`](https://huggingface.co/datasets/dan5433/southgrid-assets)

下载地址：

- [任务 2 检查点 `9999.zip`](https://huggingface.co/datasets/dan5433/southgrid-assets/resolve/main/task2/9999.zip)
- [RTAB-Map 数据库 `rtabmap.db`](https://huggingface.co/datasets/dan5433/southgrid-assets/resolve/main/task1_opensource/data/world_anchored_rtabmap_20260821T195241%2B0800/rtabmap.db)

下载后，将文件放回上述 URL 路径对应的本地路径即可。RTAB-Map 数据库的正确本地路径是：

```text
task1_opensource/data/world_anchored_rtabmap_20260821T195241+0800/rtabmap.db
```

例如，使用 Hugging Face CLI 下载整个数据集：

```bash
hf download dan5433/southgrid-assets \
  --repo-type dataset \
  --local-dir .
```

其中 `9999.zip` 约 9.4 GB，`rtabmap.db` 约 360 MB。GitHub 提交中通过 `.gitignore` 排除这两个文件，以遵守 GitHub 单文件大小限制。
