"""int8 static quantization via onnxruntime."""
from pathlib import Path
from typing import Iterable

import numpy as np

from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class _NpyReader(CalibrationDataReader):
    def __init__(self, samples: Iterable[np.ndarray], input_name: str = "x"):
        self._it = iter({"__s": s} for s in samples)
        self._name = input_name
        self._samples = list(samples)
        self._cursor = 0

    def get_next(self):
        if self._cursor >= len(self._samples):
            return None
        s = self._samples[self._cursor].astype(np.float32)
        self._cursor += 1
        return {self._name: s}


def quantize(fp32_path: str | Path, int8_path: str | Path, calib: Iterable[np.ndarray]) -> Path:
    int8_path = Path(int8_path)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=_NpyReader(list(calib)),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return int8_path
