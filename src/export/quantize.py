"""int8 quantization via onnxruntime. static preferred, dynamic as fallback."""
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _static(fp32_path, int8_path, samples, input_name="x"):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self, s):
            self._s = list(s)
            self._i = 0

        def get_next(self):
            if self._i >= len(self._s):
                return None
            out = {input_name: self._s[self._i].astype(np.float32)}
            self._i += 1
            return out

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=_Reader(samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def _dynamic(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)


def quantize(fp32_path: str | Path, int8_path: str | Path, calib: Iterable[np.ndarray] | None = None) -> Path:
    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    if calib:
        try:
            _static(fp32_path, int8_path, calib)
            return int8_path
        except Exception:
            pass
    _dynamic(fp32_path, int8_path)
    return int8_path
