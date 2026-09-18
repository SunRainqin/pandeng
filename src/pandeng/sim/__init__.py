"""闭环仿真: 仅在离线验证控制与调度时使用, 不作为指标验收依据。"""

from __future__ import annotations

from .simulator import ScriptedDetector, SimulatedScenario, SimulatedSource

__all__ = ["ScriptedDetector", "SimulatedScenario", "SimulatedSource"]
