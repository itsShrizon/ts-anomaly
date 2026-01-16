"""thin wrapper over onnxruntime InferenceSession."""
import numpy as np
import onnxruntime as ort


class OrtScorer:
    def __init__(self, path: str, providers: list[str] | None = None):
        self.sess = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self.sess.run(None, {self.iname: x})[0].reshape(-1)
