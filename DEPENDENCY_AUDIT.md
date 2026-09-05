# MetaTTT dependency audit（当前工作树）

## 依赖层级与体量

| 层级 | 当前内容 | 源机体量 | 本包状态 |
|---|---|---:|---|
| 项目代码 | `src/bfa` + 当前相关脚本 | 约 0.5 MB + 约 6 MB 脚本目录 | 已复制 |
| 正式外部实现 | `NeuroTTT/CBraMod`，含 `chbmit_groupkfold`、`models` | 约 23 MB（含上游权重） | 代码已 vendored；权重不打包 |
| 项目内 CBraMod | `third_party/CBraMod` | 约 27 MB（含上游权重） | 代码已复制；权重不打包 |
| 输入清单 | windows/recordings/seizures/group-kfold | 约 7.1 MB | 已复制 |
| 当前正式结果 | `meta-ttt-chbmit-5fold-v1` | 约 344 MB | 已复制 |
| Python 环境 | `.venv` | 约 8.2 GB | 不复制，按 lock 安装 |
| 预处理 EEG cache | `bfa_cache_v3_official_noclip/cbramod` | 约 42 GB | 不复制，单独 rsync |
| 原始 CHB-MIT EDF | `chbmit-1.0.0` | 约 43 GB | 不复制；仅重建 cache 时需要 |
| 训练后 checkpoint | 15 个 MetaTTT `best.pt` | 约 12 GB | 按要求不复制 |

压缩包本身约 135.6 MB。解压目录的未压缩文件总量约 377.6 MB，主要是已经生成的
probability parquet 和结果元数据。

## 正式 MetaTTT 的真实代码依赖

正式训练/评估入口位于：

```text
external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_train.py
external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_evaluate.py
external/NeuroTTT_CBraMod/chbmit_groupkfold/meta_train.py
external/NeuroTTT_CBraMod/chbmit_groupkfold/meta_evaluate.py
external/NeuroTTT_CBraMod/chbmit_groupkfold/meta_model.py
external/NeuroTTT_CBraMod/chbmit_groupkfold/data.py
external/NeuroTTT_CBraMod/models/cbramod.py
external/NeuroTTT_CBraMod/models/criss_cross_transformer.py
external/NeuroTTT_CBraMod/models/temporal.py
```

模型导入链是 `chbmit_groupkfold.meta_* -> models.* -> einops/torch`，数据链是
`data.py -> numpy/pandas/pyarrow/torch.utils.data`，评分链是
`meta_evaluate.py -> scikit-learn + project/src/bfa/evaluation`。当前实现不依赖原始
EDF 读取；它直接从 16 通道、200 Hz、10 秒窗口的预计算 `.npy` cache 读取数据。

## Python 依赖

直接依赖已固定在 `environment/requirements_meta_ttt.txt`：

* PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、torchaudio 2.11.0+cu128：单独从
  PyTorch cu128 index 安装；
* numpy/pandas/scipy/pyarrow/mne：数组、表格、EDF/预处理兼容；
* scikit-learn/scikit-learn-extra：分类指标和现有项目兼容；
* einops：CBraMod 张量重排；
* hydra/omegaconf/pydantic/rich/typer/tqdm：当前项目脚本配置、CLI 和日志；
* matplotlib/seaborn/networkx/statsmodels/catboost/torch-geometric/zarr：保留当前项目
  的兼容运行环境；不是每个 MetaTTT 单次命令都会导入。

完整源机 freeze 在 `environment/versions.txt`，用于审计，不要用它覆盖目标机已有
CUDA/NVIDIA wheel。

## 外部非 Python 依赖

* WSL2 Ubuntu 24.04 或原生 Ubuntu 24.04；
* `python3.11-venv`、`build-essential`、`git-lfs`、`curl`、`rsync`；
* NVIDIA 驱动能够运行 CUDA 12.8 runtime；完整 CUDA toolkit 通常不是必须的；
* 约 45 GB 的预处理 cache 目标磁盘空间（实际数据约 42 GB），以及结果/临时文件余量。

## 预训练权重下载

`pretrained_weights.pth` 是外部模型资产，不在本包内。下载地址和校验值：

```bash
curl -L --fail --retry 3 \
  -o "$NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" \
  https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
echo "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178  $NEUROTTT_CODE_ROOT/pretrained_weights/pretrained_weights.pth" | sha256sum -c -
```

上游文件页面给出的文件大小约 19.8 MB，SHA-256 与当前源机记录一致：
[Hugging Face pretrained_weights.pth](https://huggingface.co/weighting666/CBraMod/blob/main/pretrained_weights.pth)。

## 当前训练瓶颈

源机正式 Meta-Band 训练的 fold 时间约为：fold 0 10.28 h、fold 1 6.72 h、fold 2
2.54 h、fold 3 2.43 h、fold 4 2.41 h；峰值显存约 9.3 GiB。代码层面最重的是：

1. `autograd.grad(..., create_graph=True, retain_graph=True)` 形成 exact second-order
   outer graph；
2. CUDA 上强制 SDPA `MATH` backend，以获得二阶导支持，牺牲 FlashAttention 路径；
3. `batch_size=32`、`effective_batch=128` 意味着一次 outer update 要做 4 个可微
   micro-step；
4. 每个 epoch 对完整 validation 做 frozen 和 per-sample adapted 两次扫描，fold 0
   有 540,350 个 validation rows；
5. 每个 epoch 保存约 832 MB 的 checkpoint，optimizer/scheduler/history/RNG 也被写入。

所以换卡不是唯一解。优先改 validation 频率/抽样 early-stop、checkpoint 写入频率和
I/O，再考虑扩大 batch；如果改成 first-order，只能作为加速 sweep，不能直接当作当前
exact MetaTTT 结果。

## 硬件判断

* 一张卡的均衡推荐：**A100 80 GB**。当前单 job 在 RTX 5090 32 GB 上已运行，记录峰值
  约 9.3 GiB，因此显存不是单 job 的硬瓶颈；A100 80 GB 的价值在于并行 fold、batch
  余量和稳定性。
* 预算优先/追求吞吐：**H100 80 GB 或 H100 NVL 94 GB**。官方规格给出 H100 SXM 80 GB、
  H100 NVL 94 GB；但当前 exact second-order 路径使用 math SDPA，不能假设按 H100
  宣传 Tensor Core 峰值等比例加速。
* A100 40 GB 可以跑单 job，但不建议作为长期迁移目标；RTX 5090 32 GB 仍可保留做
  单 job 或小规模验证。
