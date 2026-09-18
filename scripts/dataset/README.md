# Brackish 数据集准备

本目录是数据集准备的**权威副本**：`Brackish` 公开数据只有原始视频与 AAU CSV 标注，
两个 `dataset_*` 目录都由这里的脚本生成。把它们放在项目内，是为了在数据尚未下载时
就能拿到完整的准备流程与类别定义，而不是等数据到位后再去翻数据目录。

```text
scripts/dataset/
├── Brackish.names               # 6 个类别(与 dataset_yolo/data.yaml 同源)
├── prepare_yolo_dataset.py      # 策略一: YOLO 检测训练集
├── prepare_tracking_dataset.py  # 策略二: 连续视频跟踪评估集
└── README.md
```

## 0. 前置条件与源数据布局

需要 `ffmpeg` / `ffprobe`（脚本调用它们抽帧与读取视频尺寸）：

```bash
ffmpeg -version && ffprobe -version
```

脚本假设 Brackish 数据集根目录存在以下结构（`--root` 指向它）：

```text
<brackish-root>/
├── annotations/annotations_AAU/*.csv   # 官方 AAU 逐帧 CSV(分隔符 ';')
├── dataset/videos/<类别>/*.avi          # 原始 1920x1080 AVI
└── scripts/Brackish.names              # 可缺省, 缺省时回退到本目录的同名文件
```

视频目录按内容分五类：`crab`、`fish-big`、`fish-school`、`fish-small-shrimp`、
`jellyfish`。

## 1. 运行

两个脚本的 `--root` 默认值是 `Path(__file__).resolve().parents[1]`，因此**从项目内
运行时必须显式指定数据集根目录**：

```bash
cd /data2/zyq/project/pandeng

ROOT=/data2/zyq/datasets/brackish-dataset

# 策略一: YOLO 检测训练集(89 个源视频, ~12k 帧, 需要较长时间)
python scripts/dataset/prepare_yolo_dataset.py --root "$ROOT"

# 策略二: 连续视频跟踪评估集(只处理 val/test, 29 个源视频, 4965 帧)
python scripts/dataset/prepare_tracking_dataset.py --root "$ROOT"

# 也可以把产物写到别处
python scripts/dataset/prepare_yolo_dataset.py --root "$ROOT" --output /data2/zyq/datasets/dataset_yolo
```

> 策略二依赖策略一：它 `import prepare_yolo_dataset` 复用同一套标签转换函数。
> 两者必须放在同一目录下。脚本会拒绝把产物写进数据集的 `dataset/` 目录内。

生成后，检测训练用 `dataset_yolo/data.yaml`；跟踪评估用
`--type sequence_dir --source "$ROOT/dataset_tracking"`（见项目 README
「多序列数据集回放」一节）。

---

以下为原始策略说明。

## 策略一：YOLO 检测训练集

**脚本：** [prepare_yolo_dataset.py](./prepare_yolo_dataset.py)

**目的：** 为 YOLO11n 微调提供图像和检测标签。

处理流程：

1. 合并 `annotations/annotations_AAU/` 下的 CSV；
2. 按图像名、类别和框坐标去重；
3. 根据帧名还原源视频；
4. 按源视频而不是按帧划分 train/val/test；
5. 只抽取 CSV 中出现过的有标注帧；
6. 将视频帧缩放到 `960×540`；
7. 先在 AAU/COCO 的 `960×540` 标签坐标系中裁剪框；
8. 写入 YOLO `class x_center y_center width height` 标签。

输出：

```text
dataset_yolo/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
├── video_splits.json
└── video_splits.txt
```

`video_splits.txt` 按 `train`、`val`、`test` 列出实际使用的源视频文件；`video_splits.json` 是同一清单的机器可读版本，并附带源视频类别、原始尺寸和标注帧数。

### 目录内容

- `images/<split>/`：从原始 AVI 视频抽取并缩放到 `960×540` 的 PNG 帧。
- `labels/<split>/`：与图像同名的 YOLO 检测标注文件。
- `data.yaml`：YOLO 训练配置。
- `video_splits.txt`：按 `train`、`val`、`test` 列出源视频、类别和标注帧数，便于人工核对。
- `video_splits.json`：视频划分清单的机器可读版本，包含源视频路径、划分、原始尺寸和标注帧数。
- 每个图像文件都有一个同名标签文件；标签文件可以包含多行目标框。
- 当前训练数据只收录原始 CSV 中有标注的帧，不额外生成无目标负样本。

### 类别与标签格式

类别编号来自 [Brackish.names](./Brackish.names)，从 `0` 开始：

| class id | 类别       |
| -------: | ---------- |
|        0 | fish       |
|        1 | small_fish |
|        2 | crab       |
|        3 | shrimp     |
|        4 | jellyfish  |
|        5 | starfish   |

每行标签采用 YOLO 格式：

```text
class_id x_center y_center width height
```

注意：AVI 原始视频为 `1920×1080`，但 `frameExtractor.py` 和原始 COCO 元数据表明，
AAU CSV 的标注坐标系是项目预先缩放后的 `960×540` 帧坐标。生成脚本因此直接在
`960×540` 标签坐标系中裁剪和归一化，不能再按 `1920×1080` 对标签缩小一次。当前
输出图像也是 `960×540`，所以标签框与图像坐标一一对应。`Object ID` 不写入 YOLO
标签。

标签转换公式如下：

```text
x_center = (left + right) / (2 × 960)
y_center = (top + bottom) / (2 × 540)
width    = (right - left) / 960
height   = (bottom - top) / 540
```

超出 `960×540` 边界的坐标会先裁剪到图像边界；检查发现原始标注中有 744 个框存在边界超出情况。宽度或高度裁剪后不大于零的框会直接报错，不会写入无效标签。

### 当前实际划分

已生成数据的统计如下：

| 划分  | 源视频数 | 图像帧数 | 目标框数 |
| ----- | -------: | -------: | -------: |
| train |       60 |    8,775 |   23,982 |
| val   |       13 |    1,378 |    2,199 |
| test  |       16 |    2,291 |    9,384 |
| 合计  |       89 |   12,444 |   35,565 |

帧数不会严格按 `70%/15%/15%` 分配，因为实际划分单位是完整源视频。

脚本会更新 `images/`、`labels/`、`data.yaml` 和视频划分清单，不会删除上层的说明文件。如果需要重新生成，建议只清理脚本生成的内容：

```bash
rm -rf dataset_yolo/images dataset_yolo/labels
rm -f dataset_yolo/data.yaml dataset_yolo/video_splits.json dataset_yolo/video_splits.txt
python scripts/dataset/prepare_yolo_dataset.py --root "$ROOT"
```

该策略适合检测器训练，但不保留无标注帧，因此不应直接用于跟踪器的连续视频评估。

## 策略二：连续视频跟踪评估集

**脚本：** [prepare_tracking_dataset.py](./prepare_tracking_dataset.py)

**目的：** 为 YOLO + ByteTrack 等跟踪链路保留完整时间顺序。

处理流程：

1. 使用与策略一相同的源视频划分；
2. 只处理 `val` 和 `test` 源视频；
3. 对每个源视频抽取全部连续帧，而不是只抽取有标注的帧；
4. 帧仍统一缩放到 `960×540`；
5. 每一帧都创建同名标签文件；
6. 有人工标注的帧写入 YOLO 框，没有 CSV 标注的帧写入空标签；
7. 每个源视频单独存放，不能跨视频连接轨迹；
8. 复用检测训练集的标签转换函数，直接使用 `960×540` 标注坐标，不再次按 `1920×1080` 缩放；
9. 用 `sequences.json` 记录源视频、帧数、标注帧数和尺寸。

输出：

```text
dataset_tracking/
├── images/
│   ├── val/<video_stem>/
│   └── test/<video_stem>/
├── labels/
│   ├── val/<video_stem>/
│   └── test/<video_stem>/
├── classes.json
├── sequences.json
└── video_splits.txt
```

跟踪评估时应按 `sequences.json` 逐个源视频初始化和结束跟踪器，不能把不同源视频拼接成一个长序列。空标签只表示“该帧没有 CSV 人工框”，不等价于绝对不存在目标；使用这些帧进行指标计算时应明确漏标和遮挡的处理规则（项目内由 `metrics.py::DetectionEvaluator` 落地：空标签帧完全不计入 TP/FP/FN）。

`video_splits.txt` 列出跟踪评估集中的 `val` 和 `test` 源视频，以及每个视频实际抽取的总帧数和有标注帧数。三部分完整划分以 `dataset_yolo/video_splits.txt` 或 `dataset_yolo/video_splits.json` 为准。

当前已生成的跟踪评估集包含：

| 项目          |  数量 |
| ------------- | ----: |
| 连续源视频    |    29 |
| 连续图像帧    | 4,965 |
| 有 CSV 标注帧 | 3,669 |
| 空标签帧      | 1,296 |

当前跟踪数据仍不包含人工轨迹 ID。原始 CSV 的 `Object ID` 没有写入 YOLO 标签，因此可以用于连续视频输入、检测框逐帧比较和 ByteTrack 运行测试，但不能直接计算 IDF1、ID Switch 或 HOTA 等身份关联指标。若需要完整跟踪评价，应另行生成保留 `Object ID` 的 MOT 格式标注。

## 两种策略的关系

| 项目         | YOLO 训练集    | 连续跟踪评估集                            |
| ------------ | -------------- | ----------------------------------------- |
| 主要用途     | 检测器微调     | 检测-跟踪链路评估                         |
| 覆盖帧       | 仅有标注帧     | 源视频全部连续帧                          |
| 无目标帧     | 不保留         | 保留并生成空标签                          |
| 组织方式     | 按 split 汇总  | 按 split/源视频分目录                     |
| 是否保留时序 | 不作为训练输入 | 必须保留                                  |
| 标签格式     | YOLO 检测框    | YOLO 检测框，逐帧对齐                     |
| 轨迹 ID      | 不包含         | 当前也不包含，需要另行生成或使用 MOT 标注 |

## 与项目其余部分的关系

| 产物                        | 消费方                                                            |
| --------------------------- | ----------------------------------------------------------------- |
| `dataset_yolo/data.yaml`    | `scripts/train_yolo_fish.py`（YOLO11n 两阶段微调）                 |
| `dataset_yolo/`             | 检测精度评估、ONNX/TensorRT 导出的精度校验                         |
| `dataset_tracking/`         | `--type sequence_dir` 多序列回放，逐序列输出检测 P/R/F1 与跟踪异常 |
