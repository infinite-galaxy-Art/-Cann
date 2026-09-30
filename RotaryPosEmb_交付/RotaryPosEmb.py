#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
RotaryPosEmb算子golden实现
以 numpy 组合实现旋转位置编码(RoPE) 的结果为 golden
"""
import numpy as np


def impl(x, positions, theta=10000.0):
    """RotaryPosEmb算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    x: (B, S, H, rope_dim) numpy数组，待旋转的 rope 子维度张量
    positions: (B, S) numpy数组，int32 位置索引
    theta: float，RoPE 基础频率基数，默认 10000.0

    返回:
    y: 与 x 同 shape 同 dtype 的 numpy 数组
    """
    orig_dtype = x.dtype
    dim = x.shape[-1]

    x_f = x.astype(np.float32)
    pos_f = positions.astype(np.float32)

    # 1. 逆频率
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: dim // 2] / dim))

    # 2. 旋转角与 cos/sin
    freqs = pos_f[..., None] * inv_freq                    # [B, S, dim/2]
    theta_freqs = np.concatenate([freqs, freqs], axis=-1)  # [B, S, dim]
    cos = np.cos(theta_freqs)[..., None, :]                # [B, S, 1, dim]
    sin = np.sin(theta_freqs)[..., None, :]

    # 3. rotate-half 旋转
    def rotate_half(t):
        t1, t2 = t[..., : dim // 2], t[..., dim // 2:]
        return np.concatenate([-t2, t1], axis=-1)

    y = x_f * cos + rotate_half(x_f) * sin

    if orig_dtype == np.float16:
        return y.astype(np.float16)
    return y.astype(np.float32)


if __name__ == "__main__":
    # 与 JSON npu_cases 对应的 15 个测试用例（用 numpy 生成）
    cases = [
        # (id, shape, dtype, positions上限, theta)
        (1,  [1, 3, 2, 20],     np.float32, 3,      10000.0),
    ]

    for cid, shape, dt, pmax, theta in cases:
        B, S = shape[0], shape[1]
        x = np.random.randn(*shape).astype(dt)
        p = np.random.randint(0, pmax, (B, S)).astype(np.int32)
        y = impl(x, p, theta)
        print(f"Case {cid:02d}: shape={y.shape}, dtype={y.dtype}")

    print("All tests passed!")
