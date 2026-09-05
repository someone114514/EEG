# MetaTTT migration bundle

本压缩包冻结的是 **2026-09-05 当前工作树**，用于在另一台带 NVIDIA GPU 的
WSL2/Linux 设备上复现 CHB-MIT MetaTTT（以及当前 Band-TTT v2 相关脚本）。
它不是重新训练出来的精简 demo：代码、外部 `NeuroTTT/CBraMod`、清单、预训练权重下载说明和
当前已完成的 MetaTTT 结果都来自当前磁盘上的实际文件。训练产生的 `best.pt`/`last.pt`
以及 CBraMod 预训练权重按要求不放入本包；需要复现时按下方命令下载预训练权重，再使用
本包中的源代码和外部 cache 重新训练。

## 包含什么

* `project/src/bfa/`：项目本地 Python 包。
* `project/scripts/`：当前 MetaTTT、联合 TTT、评估、审计和 Band-TTT v2 相关脚本。
* `external/NeuroTTT_CBraMod/`：当前正式 MetaTTT 实现所在的外部代码树，包含
  `chbmit_groupkfold`、`models`、预训练权重说明和启动脚本。它已被 vendored，
  迁移后不需要再从 GitHub 猜版本或重新 clone。
* `project/third_party/CBraMod/`：项目内现有 CBraMod 依赖，供旧版联合 TTT 和兼容脚本使用。
* `manifests/`：窗口、recording、seizure、五折患者划分等输入清单。
* `external/NeuroTTT_CBraMod/pretrained_weights/README.md`：上游权重说明；实际
  `pretrained_weights.pth` 不打包，按下面的官方下载命令取得。
* `results/`：当前正式 release 的 validation/test probability、指标、summary、日志、运行
  manifest/history 和
  freeze/preflight 元数据；用于核对迁移后的输出是否一致。
* `environment/`：精确版本记录、PyTorch CUDA 12.8 安装方式、环境激活和预检脚本。
* `tools/`：路径设置、外部数据复制/校验和 bundle 校验脚本。

## 刻意没有塞进压缩包的内容

当前 `.venv` 约 8.2 GB；已预处理的 CHB-MIT CBraMod cache 约 42 GB；原始 CHB-MIT
数据约 43 GB。它们不适合和代码/结果重复打进一个压缩包；
`tools/verify_external_data.sh` 提供外部数据校验方式，下面也有复制命令。

本包不带训练后的 MetaTTT checkpoint。若要在另一台机器上重新训练/重新评估，必须额外迁移：

1. `bfa_cache_v3_official_noclip/cbramod`（约 42 GB）；
2. 本包中的 `manifests/`（已包含）；
3. 如果要重新生成 cache 或从原始 EDF 重做 preprocessing，再额外迁移
   `chbmit-1.0.0`（约 43 GB）。

已有 cache 和下载好的 CBraMod 预训练权重时不需要原始 EDF 才能运行 MetaTTT 训练/评估；原始数据只在重建 cache、
改变采样/通道处理或做原始数据审计时需要。若只查看/汇报本次已完成结果，则
`results/` 已包含结果文件，不需要再次运行评估。

## 推荐迁移环境

推荐目标是 **WSL2 Ubuntu 24.04 或原生 Ubuntu 24.04 + NVIDIA driver**。当前源机实测：

* Python 3.11.15；
* PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、torchaudio 2.11.0+cu128；
* CUDA runtime 12.8；
* RTX 5090 32 GB；
* BF16 可用；MetaTTT 正式训练的记录峰值显存约 9481 MiB。

显卡驱动只需要能够运行 CUDA 12.8 wheel；通常不必另外安装完整 CUDA toolkit。
如果目标机是 Windows，请先安装 WSL2/Ubuntu，再在 WSL 内执行下面命令，不要把
`C:\...` 路径直接传给 Python。

## 安装步骤

在 WSL/Linux 中进入本 bundle 根目录（也就是本 README 所在目录）：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential python3.11 python3.11-venv python3-pip

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 必须先从 PyTorch cu128 index 安装带 CUDA 的 wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128

# 再安装 MetaTTT 的 Python 依赖
python -m pip install -r environment/requirements_meta_ttt.txt
```

也可以直接运行：

```bash
bash environment/bootstrap_wsl.sh
```

`environment/versions.txt` 是源机的完整 freeze，仅用于审计/锁版本；不建议把其中的
所有 NVIDIA wheel 直接在另一台机器上盲目覆盖安装。`requirements_meta_ttt.txt` 已把
MetaTTT 运行时需要的直接包固定下来，CUDA PyTorch 单独使用上面的官方 index 安装。

### 下载 CBraMod 预训练权重

本包不携带 `pretrained_weights.pth`。从上游 Hugging Face 仓库下载到 external tree
指定的位置：

```bash
mkdir -p "$NEUROTTT_CODE_ROOT/pretrained_weights"
curl -L --fail --retry 3 \
  -o "$NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" \
  https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
echo "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178  $NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" | sha256sum -c -
```

或使用 Hugging Face CLI：

```bash
python -m pip install -U huggingface_hub
huggingface-cli download weighting666/CBraMod pretrained_weights.pth \
  --local-dir "$NEUROTTT_CODE_ROOT/pretrained_weights" \
  --local-dir-use-symlinks False
```

官方文件页面显示该文件约 19.8 MB，SHA-256 为上面锁定的值；下载后校验必须通过。

外部代码的安装方式有两种，优先第一种：

1. **推荐：使用本包内的 vendored 代码**。`environment/activate.sh` 会将
   `external/NeuroTTT_CBraMod` 和 `project/src` 放进 `PYTHONPATH`，不依赖外网 Git
   仓库版本。
2. 若要作为 Python editable package 安装，可执行
   `python -m pip install -e external/NeuroTTT_CBraMod`；但仍需保留
   `PYTHONPATH`，因为正式脚本直接导入顶层 `models` 和 `chbmit_groupkfold`。

不要用一个“最新 CBraMod”替换本包中的 external tree；那会改变模型结构、权重解释或
正式结果的代码 provenance。

## 设置数据路径

激活环境时设置三个路径。路径可以是任意绝对路径，不要求和源机盘符相同：

```bash
export BFA_ROOT="$(pwd)"
export BFA_CACHE_ROOT="/data/EEGData/bfa_cache_v3_official_noclip/cbramod"
export BFA_RAW_ROOT="/data/EEGData/chbmit-1.0.0"
source environment/activate.sh
```

`BFA_RAW_ROOT` 仅为 preprocessing/审计保留；当前 MetaTTT 的 WindowDataset 直接读取
`BFA_CACHE_ROOT` 下已经按 CBraMod 输入单位缩放好的 `.npy`。

从源机复制预处理 cache（示例，目标目录需先在目标机创建）：

```bash
mkdir -p "$BFA_CACHE_ROOT"
rsync -a --info=progress2 \
  /mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod/ \
  "$BFA_CACHE_ROOT/"
```

如果使用 Windows 共享目录，把上面的源/目标替换成各自 WSL 内的 `/mnt/<drive>/...`
路径。不要用文件名排序代替清单中的患者/recording 顺序。

校验外部数据：

```bash
bash tools/verify_external_data.sh
```

## 先做 smoke test

```bash
bash environment/preflight.sh
python -c "import torch, pandas, pyarrow, mne; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_smoke.py \
  --output-root "$BFA_ROOT/outputs/smoke" \
  --windows "$BFA_ROOT/manifests/windows.parquet" \
  --fold-root "$BFA_ROOT/manifests/groupkfold_cv_v1" \
  --cache-root "$BFA_CACHE_ROOT" \
  --pretrained "$NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" \
  --device cuda --updates 2 --batch-size 4 --workers 2
```

若只是核对已有结果，先不要写入正式 release 目录，使用新的 `--output-root`。

## 复现 MetaTTT

正式实现的入口是 `external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_train.py`，不是
项目内早期的 `scripts/212_meta_ttt_train.py`。后者仍被保留用于旧版联合 TTT 对照。

单个 fold 的训练命令示例：

```bash
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_train.py \
  --condition meta_band --fold 0 --seed 3407 \
  --output-root "$BFA_ROOT/outputs/reports/reproduction" \
  --windows "$BFA_ROOT/manifests/windows.parquet" \
  --fold-root "$BFA_ROOT/manifests/groupkfold_cv_v1" \
  --cache-root "$BFA_CACHE_ROOT" \
  --pretrained "$NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" \
  --batch-size 32 --effective-batch 128 --eval-batch-size 256 \
  --workers 8 --epochs 50 --minimum-epochs 5 --patience 7 \
  --min-delta 0.002 --lr 1e-4 --weight-decay 0.05 \
  --initial-alpha 1e-4 --device cuda
```

当前正式训练使用 exact second-order Meta-TTT：inner update 只更新最后两个
Transformer blocks 和对应 SSL head，`create_graph=True`，并强制 CUDA SDPA math backend。
这不是普通 inference-time 一阶 TTT，预计会慢很多；详见本文末尾的瓶颈分析。

如果另行提供某个训练后的 checkpoint，评估命令示例：

```bash
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_evaluate.py \
  --condition meta_band --fold 0 --split validation --seed 3407 \
  --output-root "$BFA_ROOT/outputs/reports/reproduction" \
  --windows "$BFA_ROOT/manifests/windows.parquet" \
  --fold-root "$BFA_ROOT/manifests/groupkfold_cv_v1" \
  --cache-root "$BFA_CACHE_ROOT" \
  --pretrained "$NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" \
  --recordings "$BFA_ROOT/manifests/recordings.parquet" \
  --seizures "$BFA_ROOT/manifests/seizures.parquet" \
  --batch-size 64 --workers 8
```

validation lock 成功后，才允许对 test 使用同一个 checkpoint/阈值：

```bash
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_evaluate.py \
  --condition meta_band --fold 0 --split test --allow-test --seed 3407 \
  --output-root "$BFA_ROOT/outputs/reproduction" \
  --windows "$BFA_ROOT/manifests/windows.parquet" \
  --fold-root "$BFA_ROOT/manifests/groupkfold_cv_v1" \
  --cache-root "$BFA_CACHE_ROOT" \
  --pretrained "$NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" \
  --recordings "$BFA_ROOT/manifests/recordings.parquet" \
  --seizures "$BFA_ROOT/manifests/seizures.parquet" \
  --batch-size 64 --workers 8
```

正式脚本会拒绝已经存在的 probability 文件，避免不小心覆盖结果；请用新的 output
目录或先确认目标目录是本次复现专用目录。

## 当前训练瓶颈结论

瓶颈不是单纯“显卡显存不够”。源机 Meta-Band 五折的实际记录为：fold 0 约
10.28 h、fold 1 约 6.72 h、fold 2 约 2.54 h、fold 3 约 2.43 h、fold 4 约 2.41 h；
训练峰值显存约 9.3 GiB。主要耗时来自：

1. 每个 micro-batch 都要构造可微 inner update，并通过 outer classification loss
   反传二阶图（`autograd.grad(create_graph=True, retain_graph=True)`）。
2. CUDA 上为了支持二阶导，代码强制 `SDPBackend.MATH`，放弃了 FlashAttention 等更快
   的 attention kernel。
3. 一个 outer update 实际由 `batch_size=32` 和 `effective_batch=128` 的 4 个
   differentiable micro-step 组成；每个 micro-step 都有 SSL forward/backward、
   functional detector forward 和 outer backward。
4. 每个 epoch 结束都完整扫描 validation 两次：一次 frozen，一次独立 per-sample
   TTT；后者用 `torch.func.vmap(grad(...))`，fold 0 有 540,350 个 validation rows，
   因而比训练本身更容易成为长尾。
5. 每个 epoch 同时写约 832 MB 的 `best.pt`/`last.pt`（含模型、optimizer、scheduler、
   history 和 RNG 状态）。这会把计算瓶颈和磁盘 I/O 瓶颈叠加；`last.pt` 对最终 inference
   并不需要。

因此最有效的加速顺序是：减少/延后 full validation、降低 checkpoint 写入频率、把
validation adapted logits 缓存或改成抽样 early-stop 评估、再考虑扩大 micro-batch；
直接换更贵 GPU 只能部分缓解。若允许改变算法，first-order/无二阶 sweep 会快很多，
但不再是当前 exact MetaTTT protocol，不能和正式结果无条件混报。

## 显卡建议

* **推荐一张：A100 80 GB**。对这套 exact second-order 代码，显存余量、BF16 和长期
  运行稳定性比峰值宣传算力更关键；可以容纳更大的 micro-batch/并行 fold，且不必为
  32 GB 卡反复做保守切分。
* **追求最短墙钟时间：H100 80/94 GB**。它适合同时跑多个独立 fold 或把 batch 做大，
  但当前代码强制 math SDPA，无法吃满 H100 的高效 attention 路径，因此不会按理论
  Tensor Core 峰值线性加速；软件瓶颈不改时，H100 的性价比未必优于 A100 80 GB。
* **已有 RTX 5090 32 GB：可以继续跑单 job**。记录峰值约 9.3 GiB，说明单 job 不是
  显存硬墙；更值得先改 validation/checkpoint/I/O。若要在这张卡上并行多个 job，显存
  和数据/CPU contention 会很快成为限制。
* **A100 40 GB** 可以运行单 job，但不作为长期首选；如果预算允许，优先 80 GB。

## 结果和 provenance

`provenance/path_patches.diff` 记录为了跨设备运行而对默认绝对路径做的替换，算法和
模型逻辑没有改动。`results/` 保留当前已完成的 validation/test probability、指标、
summary、日志和运行元数据，便于迁移后核对结果。压缩包之外的原始数据/cache 需要按
上文的复制命令迁移，并使用 `tools/verify_external_data.sh` 校验。
