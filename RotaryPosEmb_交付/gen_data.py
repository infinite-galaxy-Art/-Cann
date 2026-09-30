import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from RotaryPosEmb import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 ---
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
x = np.random.uniform(low=-10.0, high=10.0, size=(1, 7, 1, 16)).astype(np.float32)
x.tofile("input/case0/x.bin")
positions = np.random.randint(low=int(0), high=int(50), size=(1, 7)).astype(np.int32)
positions.tofile("input/case0/positions.bin")

theta = 10000.0

os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(x, positions, theta=theta)
if golden is not None:
    golden.tofile("output/golden_case0/golden_y.bin")

print(f"Generated test data and golden output for 1 cases.")
