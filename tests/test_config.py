from src.config import load


def test_default_loads(tmp_path):
    c = load("configs/default.yaml")
    assert c.data.window > 0
    assert c.train.lr > 0
    assert c.model.hidden > 0
