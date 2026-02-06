"""thin wrapper over onnxruntime InferenceSession. no torch on inference path."""
import numpy as np
import onnxruntime as ort


class OrtScorer:
    def __init__(self, path: str, providers: list[str] | None = None, threads: int | None = None):
        opts = ort.SessionOptions()
        if threads is not None:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=opts,
                                         providers=providers or ["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self.sess.run(None, {self.iname: x})[0].reshape(-1)
