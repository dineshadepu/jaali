import pytest
import jaali


def test_gpu_available_returns_bool():
    v = jaali.gpu_available()
    assert isinstance(v, bool)


def test_gpu_backend_selection_behavior():
    if jaali.gpu_available():
        # Should not throw
        backend = "gpu"
    else:
        # Must not silently succeed
        backend = "gpu"

    if jaali.gpu_available():
        # ok to try gpu
        pass
    else:
        with pytest.raises(ValueError):
            jaali.PyLocator2D(
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0],
                [1],
                [2],
                backend="gpu"
            )


def test_available_backends():
    backends = jaali.available_backends()
    assert "serial" in backends
    assert "parallel" in backends

    if jaali.gpu_available():
        assert "gpu" in backends
    else:
        assert "gpu" not in backends
