import numpy as np
import pytest

from ficelle_client import msgpack_numpy


def test_roundtrip_uint8_image():
    img = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
    out = msgpack_numpy.unpackb(msgpack_numpy.packb({"img": img}))
    np.testing.assert_array_equal(out["img"], img)
    assert out["img"].dtype == np.uint8


def test_roundtrip_nested_dict_with_floats_strings_and_scalars():
    payload = {
        "observation.state": np.array([0.1, -0.2], dtype=np.float32),
        "task": "pick up the mustard bottle",
        "reply": {"actions": np.zeros((10, 2), dtype=np.float32)},
        "n": 3,
        "fps": np.float64(30.0),
    }
    out = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    assert out["task"] == "pick up the mustard bottle"
    assert out["n"] == 3
    assert out["fps"] == 30.0
    np.testing.assert_allclose(out["observation.state"], payload["observation.state"])
    assert out["reply"]["actions"].shape == (10, 2)
    assert out["reply"]["actions"].dtype == np.float32


def test_object_arrays_rejected():
    # object arrays are the pickle-smuggling vector msgpack-numpy upstream falls
    # back to — the vendored codec must refuse them.
    with pytest.raises(ValueError, match="Unsupported dtype"):
        msgpack_numpy.packb({"bad": np.array([object()])})


def test_packer_instance_reusable():
    packer = msgpack_numpy.Packer()
    a = {"x": np.arange(5, dtype=np.int64)}
    np.testing.assert_array_equal(msgpack_numpy.unpackb(packer.pack(a))["x"], a["x"])
    np.testing.assert_array_equal(msgpack_numpy.unpackb(packer.pack(a))["x"], a["x"])
