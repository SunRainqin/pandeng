"""单元测试: 几何与数值工具。"""

from __future__ import annotations

import numpy as np
import pytest

from pandeng.utils.geometry import (
    area_ratio,
    box_iou,
    clamp_limits,
    clip_boxes,
    deadzone,
    expand_box,
    iou_matrix,
    normalize_error,
    slew_limit,
)


def test_normalize_error_matches_spec():
    """方案第 3 节: ex = (u - W/2)/(W/2), ey = (v - H/2)/(H/2)。"""
    ex, ey = normalize_error((640.0, 360.0), 1280, 720)
    assert ex == pytest.approx(0.0)
    assert ey == pytest.approx(0.0)

    ex, ey = normalize_error((1280.0, 720.0), 1280, 720)
    assert ex == pytest.approx(1.0)
    assert ey == pytest.approx(1.0)

    ex, ey = normalize_error((0.0, 0.0), 1280, 720)
    assert ex == pytest.approx(-1.0)
    assert ey == pytest.approx(-1.0)


def test_normalize_error_clips_to_unit_range():
    ex, ey = normalize_error((5000.0, -5000.0), 1280, 720)
    assert -1.0 <= ex <= 1.0
    assert -1.0 <= ey <= 1.0


def test_box_iou_and_matrix_agree():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([5, 5, 15, 15], dtype=np.float32)
    expected = 25.0 / (100.0 + 100.0 - 25.0)
    assert box_iou(a, b) == pytest.approx(expected)
    matrix = iou_matrix(a[None, :], b[None, :])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(expected, rel=1e-5)


def test_iou_matrix_handles_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((3, 4)), np.zeros((0, 4))).shape == (3, 0)


def test_clip_boxes_orders_corners():
    boxes = np.array([[100, 100, 10, 10]], dtype=np.float32)
    out = clip_boxes(boxes, 50, 50)
    assert out[0, 0] <= out[0, 2]
    assert out[0, 1] <= out[0, 3]
    assert out[0, 2] <= 50 and out[0, 3] <= 50


def test_area_ratio_is_fraction_of_image():
    box = np.array([0, 0, 640, 360], dtype=np.float32)
    assert area_ratio(box, 1280, 720) == pytest.approx(0.25)


def test_expand_box_keeps_center():
    box = np.array([100, 100, 200, 200], dtype=np.float32)
    out = expand_box(box, 2.0)
    assert (out[0] + out[2]) * 0.5 == pytest.approx(150.0)
    assert (out[1] + out[3]) * 0.5 == pytest.approx(150.0)
    assert out[2] - out[0] == pytest.approx(200.0)


def test_deadzone_is_continuous_and_zero_inside():
    assert deadzone(0.03, 0.05) == 0.0
    assert deadzone(-0.03, 0.05) == 0.0
    # 死区边界处输出应连续地趋近 0
    assert deadzone(0.05 + 1e-9, 0.05) == pytest.approx(0.0, abs=1e-6)
    assert deadzone(1.0, 0.05) == pytest.approx(1.0)
    assert deadzone(-1.0, 0.05) == pytest.approx(-1.0)


def test_slew_limit_respects_max_delta():
    assert slew_limit(1.0, 0.0, 0.1) == pytest.approx(0.1)
    assert slew_limit(-1.0, 0.0, 0.1) == pytest.approx(-0.1)
    assert slew_limit(0.05, 0.0, 0.1) == pytest.approx(0.05)


def test_clamp_limits_handles_reversed_bounds():
    assert clamp_limits(5.0, [1.0, -1.0]) == pytest.approx(1.0)
    assert clamp_limits(-5.0, [1.0, -1.0]) == pytest.approx(-1.0)
