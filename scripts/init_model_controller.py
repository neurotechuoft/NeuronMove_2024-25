# pip install numpy tflite-runtime

# Dataset: expecting pickle files

# Expected X shape: - (N, T, C) OR (N, T)  (we'll add channel dim if (N, T))

# Expected y shape: - (N,)


import argparse
import logging
import time
import pickle
from dataclasses import dataclass
from typing import Tuple, Optional, Any

import numpy as np
from tflite_runtime.interpreter import Interpreter


# ----------------------------
# Logger
# ----------------------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("eeg_offline_pi")

logger = setup_logger()


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    model_path: str
    data_path: str

    # Keys inside the pickle if it's a dict / might need to change to match model
    x_key: str = "X"      # EEG windows
    y_key: str = "y"      # labels

    # Run options
    max_samples: Optional[int] = 500   # None = run all
    print_every: int = 50              # log every N samples


# Loading dataset 
def load_pkl_dataset(path: str, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports pickle containing either:
      - dict with keys x_key and y_key (default "X" and "y")
      - tuple/list (X, y)

    Returns:
      X: float32, shape (N, T, C)
      y: int64, shape (N,)
    """
    with open(path, "rb") as f:
        obj: Any = pickle.load(f)

    if isinstance(obj, dict):
        if x_key not in obj or y_key not in obj:
            raise KeyError(
                f"Pickle dict must contain keys '{x_key}' and '{y_key}'. Found: {list(obj.keys())}"
            )
        X = obj[x_key]
        y = obj[y_key]
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, y = obj
    else:
        raise TypeError(
            "Unsupported pickle format. Expected dict {'X':..., 'y':...} or tuple/list (X, y)."
        )

    X = np.asarray(X)
    y = np.asarray(y)

    # Ensure X is (N, T, C)
    if X.ndim == 2:
        X = X[:, :, None]  # (N, T) -> (N, T, 1)
    elif X.ndim != 3:
        raise ValueError(f"Expected X to have 2 or 3 dims, got shape {X.shape}")

    # Ensure y is (N,)
    y = y.reshape(-1)

    return X.astype(np.float32), y.astype(np.int64)


# ----------------------------
# Preprocessing (MUST MATCH TRAINING)
# ----------------------------
def preprocess_window(x: np.ndarray) -> np.ndarray:
    
    # x: one EEG window, shape (T, C), float32
    # Need to replace with exact model preprocessing

    
    # Placeholder: per-window z-score normalization
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    x = (x - mean) / std

    return x.astype(np.float32)


# ----------------------------
# TFLite model runner
# ----------------------------
class TFLiteRunner:
    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

        logger.info(
            f"Model input:  shape={self.inp['shape']}, dtype={self.inp['dtype']}, quant={self.inp.get('quantization')}"
        )
        logger.info(
            f"Model output: shape={self.out['shape']}, dtype={self.out['dtype']}, quant={self.out.get('quantization')}"
        )

    def _quantize_if_needed(self, x: np.ndarray) -> np.ndarray:
        """
        If model expects int8 input, quantize using input scale/zero_point.
        If model expects float32, pass through.
        """
        dtype = self.inp["dtype"]
        if dtype == np.int8:
            scale, zero_point = self.inp["quantization"]
            if scale == 0:
                raise RuntimeError("Invalid quantization params: scale=0")
            xq = np.round(x / scale + zero_point).astype(np.int8)
            return xq
        return x.astype(dtype)

    def _dequantize_if_needed(self, y: np.ndarray) -> np.ndarray:
        """
        If model output is int8, dequantize to float32 for interpretation.
        """
        if self.out["dtype"] == np.int8:
            scale, zero_point = self.out["quantization"]
            return (y.astype(np.float32) - zero_point) * scale
        return y.astype(np.float32)

    def infer(self, x_batched: np.ndarray) -> np.ndarray:
        """
        x_batched: shape must match model input (usually (1, T, C) or (1, T, C, 1))
        """
        x_in = self._quantize_if_needed(x_batched)
        self.interpreter.set_tensor(self.inp["index"], x_in)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])
        return self._dequantize_if_needed(y)


# ----------------------------
# Output decoding (depends on model)
# ----------------------------
def decode_prediction(y: np.ndarray) -> int:
    """
    Common cases:
    - Softmax/logits: shape (1, K) -> argmax
    - Binary sigmoid: shape (1, 1) -> threshold 0.5
    """
    y = np.asarray(y)
    if y.size == 1:
        return int(y.flatten()[0] >= 0.5)
    return int(np.argmax(y.flatten()))

# Main evaluation loop
def run(cfg: Config):
    X, y = load_pkl_dataset(cfg.data_path, cfg.x_key, cfg.y_key)
    runner = TFLiteRunner(cfg.model_path)

    N = X.shape[0]
    n_run = N if cfg.max_samples is None else min(N, cfg.max_samples)

    expected_shape = tuple(runner.inp["shape"])  
    logger.info(f"Dataset X shape: {X.shape} (will run {n_run} samples)")
    logger.info(f"Expected model input shape: {expected_shape}")

    correct = 0
    t0 = time.time()

    for i in range(n_run):
        x_i = X[i]  # (T, C)

        # preprocess data for evaluation
        x_i = preprocess_window(x_i)

        # add batch dimension
        x_b = x_i[None, :, :]  # (1, T, C)

        # if model expects more dimensions, add them
        if len(expected_shape) == 4 and x_b.ndim == 3:
            x_b = x_b[..., None]

        if tuple(x_b.shape) != expected_shape:
            raise RuntimeError(
                f"Input shape mismatch at sample {i}: got {x_b.shape}, expected {expected_shape}. "
                "Fix preprocess/reshape to match Nicole's model input."
            )

        # inference
        y_hat_raw = runner.infer(x_b)

        # class decoding
        pred = decode_prediction(y_hat_raw)

        if pred == int(y[i]):
            correct += 1

        if (i + 1) % cfg.print_every == 0:
            acc = correct / (i + 1)
            logger.info(f"Ran {i+1}/{n_run} | acc={acc:.3f} | last_pred={pred} | last_true={int(y[i])}")

    dt = time.time() - t0
    acc = correct / n_run if n_run > 0 else 0.0
    ips = (n_run / dt) if dt > 0 else 0.0

    logger.info("Done.")
    logger.info(f"Final accuracy: {acc:.4f} ({correct}/{n_run})")
    logger.info(f"Throughput: {ips:.2f} inferences/sec on this Pi")
    logger.info(f"Total time: {dt:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to nicole_model.tflite")
    parser.add_argument("--data", required=True, help="Path to dataset .pkl containing X and y")
    parser.add_argument("--x_key", default="X", help="If pickle is a dict, key for EEG windows")
    parser.add_argument("--y_key", default="y", help="If pickle is a dict, key for labels")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--print_every", type=int, default=50)
    args = parser.parse_args()

    cfg = Config(
        model_path=args.model,
        data_path=args.data,
        x_key=args.x_key,
        y_key=args.y_key,
        max_samples=args.max_samples,
        print_every=args.print_every,
    )

    run(cfg)


if __name__ == "__main__":
    main()
