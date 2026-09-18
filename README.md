# pandeng

《04_模型训练与自主跟踪方案.md》的工程实现。首版交付范围为
**检测 + 跟踪 + DINOv3 低频校正 + 传统图像伺服控制**；时序预测为第二阶段。

```text
图像 → YOLO11n → ByteTrack → 目标状态 → 图像伺服 → 仲裁
 │                                       ↑
 └─ 每 10 个处理帧／事件触发 → DINOv3 + 目标记忆 → 关联校正（异步，快环不等待）
```

## 1. 环境

```bash
conda activate pandeng     # /data2/zyq/conda-envs/pandeng，Python 3.10 + torch 2.6.0+cu118

# 重建
conda create -n pandeng python=3.10 -y && conda activate pandeng
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt && pip install -e .
```

多卡机器上 0/1 号卡常被占用，可用 `CUDA_VISIBLE_DEVICES` 或覆盖
`detector.device` / `corrector.device`。

## 2. 资源准备

项目依赖两类外部资源，都**不随代码分发**，需要按本节准备：

| 资源            | 用途                             | 准备方式                                          |
| --------------- | -------------------------------- | ------------------------------------------------- |
| DINOv3 权重     | 慢速环外观骨干（实验组 B/C/D/E） | `scripts/download_dinov3.py` 从 ModelScope 拉取 |
| Brackish 数据集 | YOLO 微调 + 多序列跟踪评估       | `scripts/dataset/` 下的脚本从原始视频/CSV 生成  |

只有 `src/`、`configs/`、`scripts/` 是自足的：不做准备也能跑
`source.type=simulated` + `detector.backend=scenario|mock` 的链路自检（见第 3 节）。

### 2.1 DINOv3 权重

项目使用 ModelScope 上的 DINOv3 Transformers 快照。所有需要 DINOv3
的实验组统一通过 `configs/dinov3_local.yaml` 配置本地资源。

```bash
# 获取默认的 ViT-S/16 快照
python scripts/download_dinov3.py --dest /data2/zyq/weights/DINOv3

# 离线校验 / 打包传输 / 只看下载计划
python scripts/download_dinov3.py --verify
python scripts/download_dinov3.py --pack /data2/zyq/weights/dinov3_bundle.tar.gz
python scripts/download_dinov3.py --dry-run
```

产出 `<dest>/hf/<repo>/`，供 `transformers` 离线加载。下载中断后重跑即可续传。

```bash
python -m pandeng.cli.track --overlay configs/dinov3_local.yaml \
    --override source.type=simulated detector.backend=scenario
```

实际使用的骨干会写进 `summary.json`。**跑实验组 B/C/D/E 前先确认
`embedder_is_fallback` 为 `false`**——为 `true` 说明用了 torchvision 回退骨干，
关联改善结论无效。

命令行支持重复指定 `--overlay`，按从左到右的顺序合并配置：

```bash
python -m pandeng.cli.track \
    --overlay configs/experiment_b.yaml \
    --overlay configs/dinov3_local.yaml
```

A 组不加载 DINOv3；B/C/D/E 由 `scripts/run_experiments.py` 自动叠加
`configs/dinov3_local.yaml`。

### 2.2 Brackish 数据集

公开数据只有原始 AVI 视频与 AAU 逐帧 CSV 标注。两个训练/评估目录都由
`scripts/dataset/` 下的脚本生成，**注意不能用同一份数据同时充当检测训练集和
跟踪评估集**：检测训练要剔除无标注帧，而跟踪评估必须保留完整时间顺序。

脚本、类别定义与完整策略说明都在项目内：

```text
scripts/dataset/
├── README.md                    # 完整数据准备策略(含统计值与注意事项)
├── Brackish.names               # 6 个类别
├── prepare_yolo_dataset.py      # 策略一: YOLO 检测训练集
└── prepare_tracking_dataset.py  # 策略二: 连续视频跟踪评估集
```

需要 `ffmpeg` / `ffprobe`，并要求数据集根目录存在
`annotations/annotations_AAU/*.csv` 与 `dataset/videos/<类别>/*.avi`：

```bash
ROOT=/data2/zyq/datasets/brackish-dataset

# 策略一: 检测训练集(89 个源视频, 按源视频划分 train/val/test, 避免相邻帧泄漏)
python scripts/dataset/prepare_yolo_dataset.py --root "$ROOT"

# 策略二: 连续跟踪评估集(只取 val/test, 每视频一个目录, 无标注帧写空标签)
python scripts/dataset/prepare_tracking_dataset.py --root "$ROOT"
```

| 产物                  | 内容                                                | 消费方                             |
| --------------------- | --------------------------------------------------- | ---------------------------------- |
| `dataset_yolo/`     | 12,444 帧 / 35,565 框，源视频级 train/val/test 划分 | `scripts/train_yolo_fish.py`     |
| `dataset_tracking/` | 29 个序列 / 4,965 帧（其中 1,296 帧无标注）         | `--type sequence_dir` 多序列回放 |

两类数据的类别编号一致（`fish`/`small_fish`/`crab`/`shrimp`/`jellyfish`/`starfish`，
即 `0`–`5`），输出帧统一为 `960×540`。标注坐标系是 `960×540`，**不是**
AVI 的 `1920×1080`——转换时不能再缩一次，这是脚本里显式校验的一点。

细节（去重规则、边界裁剪、划分统计、空标签语义）见
[`scripts/dataset/README.md`](scripts/dataset/README.md)。

> 空标签只表示「该帧没有 CSV 人工框」，**不等于该帧没有目标**。标注密度在序列
> 之间极不均匀：29 个序列里有 14 个是逐帧全标，而最稀疏的两个序列 121 帧只标了
> 7 帧。若把空标签当作「无目标」的负样本，仅这两个序列就会凭空产生 200 多个
> 误检，因此跟踪评估里空标签帧被完全排除在检测 TP/FP/FN 之外。

## 3. 快速验证

```bash
# 闭环仿真（理想检测器 + 占位外观描述子），无需数据集与权重
python -m pandeng.cli.track --override detector.backend=scenario corrector.backend=mock \
    source.type=simulated source.max_frames=400 source.fps=20 source.realtime_pacing=true

# 视频回放（开环）
python scripts/make_demo_video.py --out data/demo/demo.mp4 --seconds 20 --fps 20
python -m pandeng.cli.track --override detector.backend=mock corrector.backend=mock \
    source.type=video source.path=data/demo/demo.mp4
```

输出在 `runs/<run_name>/`：`summary.json`（指标与成功判据）、`frames.csv`
（逐帧状态）、`events.jsonl`（校正/门控/代次/慢链路故障）、`overlay.mp4`。

## 3.1 多序列数据集回放

`dataset_tracking` 这类数据集把**每个视频单独放一个文件夹**：标注策略决定了
视频里很多帧根本没有标注，拆成散图会丢掉帧间关系。用 `source.type=sequence_dir`
可以一次跑完全部序列，并在序列边界做彻底的状态重置。

```text
<root>/images/<split>/<video>/<frame>.png
<root>/labels/<split>/<video>/<frame>.txt   # 可选, 与图像逐帧对齐, 空文件=无标注
```

```bash
# 一次跑完 val + test 的全部 29 个序列
CUDA_VISIBLE_DEVICES=1 python -m pandeng.cli.track \
    --type sequence_dir \
    --source /data2/zyq/datasets/brackish-dataset/dataset_tracking \
    --split val,test \
    --overlay configs/dinov3_local.yaml \
    --override detector.backend=ultralytics \
               detector.weights=weights/fish_yolo11n.pt \
               detector.device=cuda:0 corrector.device=cuda:0 \
    --run-name seq_all

# 先小规模试跑: 只跑 3 个序列、每个 40 帧
python -m pandeng.cli.track --type sequence_dir \
    --source /data2/zyq/datasets/brackish-dataset/dataset_tracking \
    --split test --max-sequences 3 --max-frames-per-sequence 40 --no-video
```

相关参数:

| 参数                          | 说明                                                 |
| ----------------------------- | ---------------------------------------------------- |
| `--split`                   | `val` / `test` / `val,test`, 默认两者都跑      |
| `--max-sequences`           | 最多跑几个序列(调试用)                               |
| `--max-frames-per-sequence` | **每个**序列的帧数上限(调试用)                 |
| `--max-frames`              | 整个运行的帧数上限;`sequence_dir` 下不要用它做限速 |
| `source.require_labels`     | 置`true` 可跳过完全没有标注的序列                  |

**为什么必须逐序列重置状态。** 连续性只在视频**内部**成立: 上一段视频的最后
一帧与下一段视频的第一帧毫无关系。若不重置, 轨迹 ID 会跨视频延续、外观模板会
把上一个视频里的鱼当成同一个体、目标代次也会莫名翻倍, 统计出来的「误切目标」
与「重捕时间」全是假的。因此进入新序列时以下状态全部归零:

- `ByteTracker`(轨迹 ID 从 1 重数)、`TargetTracker` + `TargetSelector`(代次归 0)
- `TemplateBank`(清空外观模板)、`ImageServoController`(PID 与死区状态)
- `ControlArbiter`(链路健康)、时序预测器
- 调度器**序列纪元**(`epoch`): 排队请求被丢弃, 上一序列仍在执行的校正结果
  即使晚到也不会进入新序列
- `frame_id` 归零、指标另起一段统计

**无标注帧不是负样本。** `dataset_tracking` 里大量帧没有标注(例如某视频
120 帧只有 35 帧带框), 把它们当成「无目标」会把正常检测计成误检。因此
`metrics.py::DetectionEvaluator` 只对**有标注的帧**做贪心 IoU 匹配统计
TP/FP/FN, 无标注帧的数量与其中的检测数单独上报(`unannotated_frames` /
`detections_on_unannotated` / `sample_coverage_ratio`)。

产物结构与单序列一致, 但**每个序列一个叠加视频** `overlay_<split>_<video>.mp4`
(绝不拼接), `frames.csv` 新增首列 `sequence`, 且 `frame_id` 逐序列归零。
`summary.json` 中的 `sequences` 为逐序列指标数组:

```json
{
  "num_sequences": 13,
  "sequences": [
    {"name": "val/2019-03-20_23-53-40to...", "frames": 238,
     "visibility_ratio": 0.82, "target_switches": 0, "losses": 6,
     "detection": {"annotated_frames": 238, "unannotated_frames": 0,
                   "tp": 300, "fp": 11, "fn": 71,
                   "precision": 0.965, "recall": 0.809, "f1": 0.880}}
  ]
}
```

命令行末尾还会打印一张逐序列速览表。

> **注意** 回放真实数据集时舵机不会真的动(帧源没有 `apply_command`),
> 目标在画面里的位置只反映视频本身的构图而非控制器效果。此时
> `central_ratio_when_visible` 衡量的是素材而不是算法,
> `summary.json` 里的 `closed_loop: false` 就是该提示。

## 4. 目录

```
configs/           default + experiment_{a..e} + dinov3_local
src/pandeng/
  types.py         快慢环数据契约
  metrics.py       第 7 节指标与成功判据
  perception/      detector / tracker / embedder / dinov3_loader / association / corrector
  memory/          template_bank（少量高置信度外观模板）
  scheduling/      慢环异步调度（限流、单次在飞、队列仅留最新）
  control/         servo / target_state / arbiter
  prediction/      时序预测接口 + 恒速基线
  io/              video_source / recorder
  sim/             闭环仿真载体（仅离线验证控制与调度，不作验收依据）
  pipeline/        主闭环装配
scripts/           download_dinov3 / train_yolo_fish
                   export_onnx / run_experiments / make_demo_video
  dataset/         Brackish 数据准备(两个策略脚本 + 类别表 + 策略说明)
tests/             115 项(涉及外部数据集的部分会自动跳过)
```

## 5. 训练与部署

```bash
# YOLO11n 两阶段微调：先适配鱼类检测头 → 再低学习率微调后部网络
python scripts/train_yolo_fish.py \
    --data /data2/zyq/datasets/brackish-dataset/dataset_yolo/data.yaml \
    --base-weights yolo11n.pt \
    --epochs-head 20 --epochs-finetune 60 --imgsz 640 --batch 16

# 端侧引擎导出（含精度校验）
python scripts/export_onnx.py --kind detector --weights weights/fish_yolo11n.pt \
    --imgsz 640 --fp16 --check --out weights/fish_yolo11n.onnx
python scripts/export_onnx.py --kind dinov3 \
    --dinov3-weights /data2/zyq/weights/DINOv3/hf/dinov3-vits16-pretrain-lvd1689m \
    --impl transformers --input-size 224 --check --out weights/dinov3_vits16.onnx
```

INT8 需用代表性水下数据校准并通过精度检查后才可采用，默认关闭。

训练/评估数据集的准备与路径见 [2.2 Brackish 数据集](#22-brackish-数据集)。

## 6. 实验组

```bash
# 仿真流程验证；B/C/D 自动加载 configs/dinov3_local.yaml
python scripts/run_experiments.py --groups A B C D --realtime \
    --override source.type=simulated detector.backend=scenario recorder.save_video=false

# Brackish 多序列数据集上的检测/链路验证
# (sequence_dir 下 --max-frames 自动解释为逐序列上限)
CUDA_VISIBLE_DEVICES=4 python scripts/run_experiments.py --groups A B C D \
    --type sequence_dir \
    --source /data2/zyq/datasets/brackish-dataset/dataset_tracking \
    --split val,test --max-frames 300 --no-record \
    --out runs/experiments/brackish_tracking \
    --override detector.backend=ultralytics \
               detector.weights=/data2/zyq/project/pandeng/weights/fish_yolo11n.pt \
               detector.device=cuda:0 corrector.device=cuda:0
```

| 组 | 配置                                 | 主要指标                   |
| -- | ------------------------------------ | -------------------------- |
| A  | YOLO11n + ByteTrack + 图像伺服       | 基线成功率、误切目标、丢失 |
| B  | A + 每十帧 DINOv3 校正（仅周期触发） | 关联改善、重捕时间、开销   |
| C  | B + DINOv3 事件触发                  | 关键时刻响应与额外推理次数 |
| D  | C + DINOv3 + 恒速预测                | 预测辅助控制基线           |
| E  | C + DINOv3 + 学习的时序预测          | 相对 D 组的跟随增益        |

A 组不构建慢速校正器；B/C/D/E 使用 `corrector.backend=dinov3`，
并自动加载 `configs/dinov3_local.yaml`。E 组还需要
`weights/temporal_head.pt`，当前学习型时序头训练流程尚未交付。

输出 `runs/experiments/<时间戳>/comparison.{csv,json}`（含均值/标准差与 D、E
相对 C 的跟随增益）。

## 7. 与方案条款的对应

| 条款                               | 实现                                         | 验证                                  |
| ---------------------------------- | -------------------------------------------- | ------------------------------------- |
| 快环 10–20Hz、校正每 10 帧        | `pipeline` / `scheduling`                | `fast_hz`、`slow_loop.request_hz` |
| 同时最多一次校正、队列仅留最新     | `scheduling/corrector_scheduler.py`        | 单测                                  |
| 快环不等待校正                     | 独立线程 + 非阻塞`poll()`                  | 单测                                  |
| 结果门控（年龄 + 目标代次）        | `perception/corrector.py::CorrectionGate`  | 单测                                  |
| 不用旧目标框覆盖当前观测           | `control/target_state.py`                  | 只取差值 + 运动对齐 + 平滑限幅        |
| 模板只在高置信度一致匹配时更新     | `memory/template_bank.py`                  | 单测（含污染防护）                    |
| 慢环不能绕过快环控制推进器         | `perception/corrector.py` 返回类型无控制量 | 单测                                  |
| 慢链路异常时回退基线，否则退出追近 | `control/arbiter.py`                       | 单测                                  |
| 死区、限幅、变化率限制             | `control/servo.py`                         | 单测                                  |
| 画面中央 50% 宽高区域判据          | `metrics.py`                               | 单测                                  |
| 每个视频独立成序列, 内部连续       | `io/video_source.py::SequenceDirSource`    | `test_sequences.py`                 |
| 跨序列不得泄漏轨迹/模板/校正结果   | `pipeline` 重置 + 调度器`epoch`          | `test_sequences.py`                 |
| 无标注帧不得计为负样本             | `metrics.py::DetectionEvaluator`           | `test_sequences.py`                 |

## 8. 已知边界

1. **仿真结果不能用于验收**。方案第 7 节要求整机实测；`sim/` 只用于验证控制律、
   调度与门控逻辑是否自洽。
2. **当前对照组无区分度**。已跑的 A~D 居中率在 0.997~1.000，因为
   `detector.backend=scenario` 使用真值框且场景中无遮挡/交叉/相似干扰目标。
   需要构造有挑战的场景后，实验组结论才有意义。
3. **Brackish 检测评估已完成，真实连续视频验收仍未完成**。当前
   `weights/fish_yolo11n.pt` 已在 Brackish 的 val/test 划分上评估，
   结果见 `runs/seq_val/summary.json` 等运行目录。`sequence_dir` 回放已经能
   逐序列给出检测质量(P/R/F1)与跟踪异常计数，但它是**开环**的：舵机不参与，
   因此不能替代连续视频上的 ByteTrack 重捕、DINOv3 关联与自主控制验收。
4. **真实数据集上的居中率不构成控制指标**。`sequence_dir` 回放不闭环，
   目标在画面中的位置只反映视频本身的构图，`central_ge_80pct` 判据不适用
   （`summary.json` 的 `closed_loop: false` 即为提示）。
5. `depth_error` 在真实艇上需要深度传感器；无传感器时为 `null`，
   `depth_within_015m` 判据会 FAIL，这是预期行为。
6. **ROS 2 未接入**。`control/arbiter.py` 是仲裁节点的占位实现，接口已按最终
   形态固定。
7. **深度调整未闭环**。首版按方案要求固定深度。
8. **时序预测为第二阶段**。`LearnedPredictor` 定义了契约与加载逻辑，
   训练流程尚未实现。

## 9. 测试

```bash
pytest tests -q     # 115 项
```

覆盖开发中实际踩到的缺陷，作为回归防护：卡尔曼增益维度、模板引导死锁、
lost 轨迹 ID 保持、感知时间戳归属、慢链路降级下的基线跟踪、周期节拍
「只能晚到不能丢失且不得加速」、DINOv3 容错权重加载与 `allow_fallback`
语义，以及多序列回放中的逐序列状态重置与「无标注 ≠ 负样本」。

`tests/test_dataset_scripts.py` 还负责把 `scripts/dataset/` 下的数据准备脚本与
已生成的数据集对齐：重新推导出的源视频划分、帧数与目标框数（`60/8775/23982`、
`13/1378/2199`、`16/2291/9384`）、逐序列标注帧数，以及 `960×540` 标签坐标系，
都必须与 `dataset_tracking/sequences.json` 和文档声明一致。没有 Brackish 源数据
的机器上这些用例会自动跳过，不影响其余测试。
