#!/usr/bin/env python
import pytest
import ionosphereAI as ia
from pathlib import Path

R = Path(__file__).parent


def test_parse_params(tmp_path):
    with pytest.raises(FileNotFoundError):
        ia.dio.get_sensor_config(tmp_path)

    pfn = R.parent / "hst0.ini"
    P = ia.dio.get_sensor_config(pfn)
    assert P.getint("main", "xpix") == 512


def test_file_read(tmp_path):
    fn = R / "data/testframes.DMCdata"

    with pytest.raises(FileNotFoundError):
        ia.dio.get_file_info(tmp_path, {})

    up = {"header_bytes": 4, "xy_pixel": (512, 512), "xy_bin": (1, 1), "twoframe": True}
    finf = ia.dio.get_file_info(fn, up)

    with pytest.raises(FileNotFoundError):
        ia.reader.get_frames(tmp_path, ifrm=0, finf=finf, up=up)

    frame = ia.reader.get_frames(fn, ifrm=0, finf=finf, up=up)
    assert frame.ndim == 3
    assert frame.shape == (2, 512, 512)


if __name__ == "__main__":
    pytest.main([__file__])
