"""闭环仿真: 虚拟艇体 + 虚拟鱼 + 虚拟相机。

存在的理由: 实验组 A~E(方案第 7 节)需要比较"跟随增益", 而跟随增益只有在
**闭环**下才有意义 —— 控制器必须能真正改变目标在画面中的位置。真实数据回放
是开环的, 无法验证这一点, 因此这里提供一个最小但物理自洽的闭环载体:

- 虚拟艇体: 一阶运动学, 由 `ControlCommand` 的 surge/yaw/heave 驱动;
- 虚拟鱼: 世界坐标系下的低速游动;
- 虚拟相机: 固定在艇体上, 把鱼的世界坐标投影为图像坐标与表观尺寸。

由此形成真实耦合:
* 转向改变目标水平位置 -> 图像伺服可以用 ex 收敛;
* 前进缩短距离 -> 目标框面积增大 -> 可验证"鱼框快速增大则减速/停止接近";
* 目标游出视场 -> 可验证"目标丢失 -> 限时搜索 -> 超时退出"。

注意: 该仿真**只用于链路与控制的离线验证**, 不作为指标验收依据。方案第 7 节
要求"指标均以整机实测验收"。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..io.video_source import FrameSource
from ..types import Detection

__all__ = ["SimulatedScenario", "SimulatedSource", "ScriptedDetector"]

LOGGER = logging.getLogger(__name__)


class SimulatedScenario:
    """世界模型: 艇体、鱼、相机。"""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fov_h_deg: float = 70.0,
        fov_v_deg: float = 45.0,
        max_yaw_rate: float = 0.6,      # rad/s
        max_speed: float = 0.8,         # m/s
        max_depth_rate: float = 0.35,   # m/s
        fish_size_m: float = 0.35,
        n_fish: int = 1,
        seed: int = 0,
        noise_std_px: float = 1.5,
        dropout_prob: float = 0.0,
        initial_distance: float = 6.0,
        initial_bearing_deg: float = 18.0,
        initial_depth_offset: float = 1.0,
    ) -> None:
        self.width, self.height = int(width), int(height)
        self.fps = float(fps)
        self.dt = 1.0 / max(1e-6, self.fps)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_speed = float(max_speed)
        self.max_depth_rate = float(max_depth_rate)
        self.fish_size_m = float(fish_size_m)
        self.n_fish = max(1, int(n_fish))
        self.noise_std_px = float(noise_std_px)
        self.dropout_prob = float(dropout_prob)
        self.rng = np.random.default_rng(int(seed))

        self.fov_h = np.deg2rad(fov_h_deg)
        self.fov_v = np.deg2rad(fov_v_deg)

        # 艇体状态: 世界坐标(x 前, y 右, z 下) 中的航向与位置
        self.vehicle_yaw = 0.0
        self.vehicle_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # 鱼: 相对艇体初始位置的极坐标
        bearing = np.deg2rad(initial_bearing_deg)
        self.fish_pos = np.array(
            [
                initial_distance * np.cos(bearing),
                initial_distance * np.sin(bearing),
                initial_depth_offset,
            ],
            dtype=np.float64,
        )
        self.fish_state = [
            {
                "pos": self.fish_pos.copy(),
                "phase": 0.0,
                "speed": 0.25 + 0.05 * i,
            }
            for i in range(self.n_fish)
        ]
        self.time = 0.0
        self.last_command = np.zeros(3, dtype=np.float64)
        self._background: Optional[np.ndarray] = None
        self._depth_reference: Optional[float] = None

    # -- 动力学 -------------------------------------------------------------
    def step(self, command: np.ndarray | None = None, dt: Optional[float] = None) -> None:
        """推进一个控制周期。"""
        dt = self.dt if dt is None else float(dt)
        if command is not None:
            self.last_command = np.asarray(command, dtype=np.float64).ravel()[:3]

        surge, yaw, heave = self.last_command

        self.vehicle_yaw += float(np.clip(yaw, -1.0, 1.0)) * self.max_yaw_rate * dt
        speed = float(np.clip(surge, -1.0, 1.0)) * self.max_speed
        self.vehicle_pos[0] += speed * np.cos(self.vehicle_yaw) * dt
        self.vehicle_pos[1] += speed * np.sin(self.vehicle_yaw) * dt
        self.vehicle_pos[2] += float(np.clip(heave, -1.0, 1.0)) * self.max_depth_rate * dt

        self.time += dt
        # 鱼在水平面内做缓慢的绕行, 并带轻微上下起伏
        for i, fish in enumerate(self.fish_state):
            phase = fish["phase"] + fish["speed"] * dt
            fish["phase"] = phase
            omega = 0.18 + 0.05 * i
            fish["pos"][0] += fish["speed"] * dt * np.cos(phase * omega)
            fish["pos"][1] += fish["speed"] * dt * np.sin(phase * omega)
            fish["pos"][2] += 0.05 * np.sin(self.time * 0.5 + i) * dt

    # -- 观测 ---------------------------------------------------------------
    def _project(self, fish_pos: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """把世界坐标投影为 (u, v, apparent_height_px)。不可见时返回 None。"""
        relative = np.asarray(fish_pos, dtype=np.float64) - self.vehicle_pos
        forward = relative[0] * np.cos(self.vehicle_yaw) + relative[1] * np.sin(self.vehicle_yaw)
        lateral = -relative[0] * np.sin(self.vehicle_yaw) + relative[1] * np.cos(self.vehicle_yaw)
        vertical = relative[2]
        if forward <= 0.3:
            return None

        bearing_h = np.arctan2(lateral, forward)
        bearing_v = np.arctan2(vertical, forward)
        if abs(bearing_h) > self.fov_h * 0.5 or abs(bearing_v) > self.fov_v * 0.5:
            return None

        u = self.width * 0.5 * (1.0 + bearing_h / (self.fov_h * 0.5))
        v = self.height * 0.5 * (1.0 + bearing_v / (self.fov_v * 0.5))

        # 小孔模型: 表观尺寸 ∝ 焦距 * 实际尺寸 / 距离
        distance = float(np.linalg.norm(relative))
        focal = self.width * 0.5 / np.tan(self.fov_h * 0.5)
        app_h = focal * self.fish_size_m / max(0.5, distance)
        return float(u), float(v), float(app_h)

    def ground_truth_boxes(self) -> List[np.ndarray]:
        """当前帧所有可见鱼的真值框(xyxy)。"""
        boxes: List[np.ndarray] = []
        for fish in self.fish_state:
            projected = self._project(fish["pos"])
            if projected is None:
                continue
            u, v, app_h = projected
            half_h = max(6.0, app_h * 0.5)
            half_w = half_h * 2.0
            u += float(self.rng.normal(0.0, self.noise_std_px))
            v += float(self.rng.normal(0.0, self.noise_std_px))
            boxes.append(
                np.array(
                    [u - half_w, v - half_h, u + half_w, v + half_h], dtype=np.float32
                )
            )
        return boxes

    def render(self, boxes: List[np.ndarray]) -> np.ndarray:
        """渲染画面。

        背景(水色渐变 + 噪声)只生成一次并复用: 1280x720 的逐帧高斯噪声会占
        掉数十毫秒, 使仿真本身成为瓶颈, 掩盖真实链路耗时。
        """
        import cv2

        if self._background is None:
            yy = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
            base = np.zeros((self.height, self.width, 3), dtype=np.float32)
            base[:, :, 0] = 125.0 - 65.0 * yy + 7.0
            base[:, :, 1] = 95.0 - 45.0 * yy + 5.0
            base[:, :, 2] = 45.0 - 18.0 * yy + 3.0
            base += self.rng.normal(0.0, 3.0, base.shape).astype(np.float32)
            self._background = np.clip(base, 0, 255).astype(np.uint8)
        frame = self._background.copy()

        # 底部纹理: 缓慢横向漂移, 给 DINOv3/直方图提供可区分背景
        for i in range(6):
            y = int(self.height * (0.72 + 0.045 * i))
            x = int((self.time * 25.0 + i * 180.0) % self.width)
            cv2.line(frame, (x - 200, y), (x + 200, y), (58, 62, 48), 2)

        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box, dtype=int)
            cv2.ellipse(
                frame,
                ((x1 + x2) // 2, (y1 + y2) // 2),
                (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)),
                0, 0, 360, (60, 180, 235), -1,
            )
            cv2.ellipse(
                frame,
                ((x1 + x2) // 2 + max(1, (x2 - x1) // 5), (y1 + y2) // 2),
                (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)),
                0, 0, 360, (25, 25, 25), -1,
            )
        return frame

    # -- 记录 ---------------------------------------------------------------
    def set_depth_reference(self, depth_m: Optional[float] = None) -> None:
        """设定深度保持的参考值; 传 None 表示以当前深度为准。"""
        self._depth_reference = (
            float(self.vehicle_pos[2]) if depth_m is None else float(depth_m)
        )

    @property
    def depth_error_m(self) -> float:
        """深度保持误差。

        固定深度阶段指令为 heave=0, 该值应接近 0; 一旦开启缓慢深度调整而
        又没有深度闭环(我们的首版没有声纳闭环), 该误差会增大 —— 这正是
        方案第 3 节要求"首轮固定深度, 稳定后再考虑深度调整"的原因。
        """
        if self._depth_reference is None:
            return 0.0
        return abs(float(self.vehicle_pos[2]) - self._depth_reference)


class SimulatedSource(FrameSource):
    """把 `SimulatedScenario` 包装成帧源, 并接受控制指令形成闭环。"""

    def __init__(
        self,
        scenario: SimulatedScenario,
        *,
        max_frames: int = 0,
        realtime_pacing: bool = True,
    ) -> None:
        self.scenario = scenario
        self.width = scenario.width
        self.height = scenario.height
        self.fps = scenario.fps
        self.max_frames = int(max_frames)
        self.realtime_pacing = bool(realtime_pacing)
        self.index = 0
        self._start: Optional[float] = None

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        import time

        if 0 < self.max_frames <= self.index:
            return None
        now = time.monotonic()
        if self.realtime_pacing:
            # 按真实帧率推进, 否则"感知频率"指标(10-20Hz)没有意义
            if self._start is None:
                self._start = now
            target = self._start + self.index / max(1e-6, self.fps)
            wait = target - now
            if wait > 0:
                time.sleep(min(wait, 1.0))
            now = time.monotonic()
        self.index += 1
        boxes = self.scenario.ground_truth_boxes()
        frame = self.scenario.render(boxes)
        return frame, now

    def apply_command(self, command) -> None:
        """接收仲裁后的控制指令并推进载体动力学。

        这是"闭环"的关键: 控制输出改变相机位姿, 从而改变下一帧目标的图像位置。
        """
        self.scenario.step(np.array([command.surge, command.yaw, command.heave], dtype=np.float64))

    def set_depth_setpoint(self, depth_m: Optional[float] = None) -> None:
        """把控制器的深度设定点转交给场景。"""
        self.scenario.set_depth_reference(depth_m)

    @property
    def depth_error_m(self) -> float:
        return self.scenario.depth_error_m

    def close(self) -> None:
        pass


class ScriptedDetector:
    """从仿真场景读取真值框的"检测器"。

    用于把**控制问题**与**检测问题**解耦: 实验组 A~E 关注的是跟随控制与
    关联校正, 使用理想检测器可以避免检测漏检掩盖控制缺陷。检测器本身的指标
    需用真实标注数据单独评估。
    """

    name = "scenario"

    def __init__(
        self,
        scenario: SimulatedScenario,
        *,
        score: float = 0.88,
        score_jitter: float = 0.05,
        dropout_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.scenario = scenario
        self.score = float(score)
        self.score_jitter = float(score_jitter)
        self.dropout_prob = float(dropout_prob)
        self.rng = np.random.default_rng(int(seed))

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.dropout_prob > 0 and self.rng.random() < self.dropout_prob:
            return []
        boxes = self.scenario.ground_truth_boxes()
        return [
            Detection(
                xyxy=np.asarray(box, dtype=np.float32),
                score=float(np.clip(self.score + self.rng.normal(0, self.score_jitter), 0.05, 0.99)),
                cls=0,
            )
            for box in boxes
        ]

    def warmup(self, image: np.ndarray, times: int = 3) -> None:
        return None

    def close(self) -> None:
        pass
