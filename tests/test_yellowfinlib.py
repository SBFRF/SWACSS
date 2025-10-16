import os
import sys
import os
import yellowfinLib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yellowfinLib as yfl
import datetime as DT
import numpy as np


def test_is_high_low_dual_freq():
    fpath = "tests/files/sonar_file_no_low.h5"
    assert yfl.is_high_low_dual_freq(fpath) == "low"
    assert yfl.is_high_low_dual_freq("file_does_exist.h5") == "high"


def test_mLabDatetime_to_epoch():
    dt = DT.datetime(1992, 12, 12, 12, 0, 0, tzinfo=DT.timezone.utc)
    python_expected = 724161600
    assert yfl.mLabDatetime_to_epoch(dt) == python_expected


@pytest.fixture
def fs():
    return 500  # sampling rate, Hz


@pytest.fixture
def order():
    return 4


def test_constant_signal_passes(fs, order):
    data = np.ones(1000)
    # cutoff anywhere—constant is all DC
    out = yfl.butter_lowpass_filter(data, cutoff=10, fs=fs, order=order)
    assert np.allclose(out, data, atol=1e-8), "DC component should be unchanged"


def test_low_freq_sine_passes(fs, order):
    t = np.arange(0, 1, 1 / fs)
    f_low = 5  # well below cutoff
    data = np.sin(2 * np.pi * f_low * t)
    out = yfl.butter_lowpass_filter(data, cutoff=20, fs=fs, order=order)
    # RMS ratio ~1
    rms_in = np.sqrt(np.mean(data**2))
    rms_out = np.sqrt(np.mean(out**2))
    assert np.isclose(rms_out, rms_in, rtol=0.05), f"Low‐freq ({f_low}Hz) should pass with little attenuation"

def test_high_freq_sine_attenuated(fs, order):
    t = np.arange(0, 1, 1 / fs)
    f_high = 200  # well above cutoff
    data = np.sin(2 * np.pi * f_high * t)
    out = yfl.butter_lowpass_filter(data, cutoff=50, fs=fs, order=order)
    # RMS ratio <<1
    rms_in = np.sqrt(np.mean(data**2))
    rms_out = np.sqrt(np.mean(out**2))
    assert rms_out < 0.1 * rms_in, f"High‐freq ({f_high}Hz) should be strongly attenuated"

def test_loadSonar_s500_binary():
    s500_folder = 'tests/files/s500_test_files'  # this is where to search for s500 files (function searches folder)
    read_out = 'tests/files/s500_test_files/test.h5'  # file name to make through read funciton
    ground_truth = 'tests/files/s500_test_files/s500_unit_test_20240528120946.h5' # file to compare as known
    yellowfinLib.loadSonar_s500_binary(s500_folder, read_out)
    generated = yellowfinLib.load_h5_to_dictionary(read_out)
    truth = yellowfinLib.load_h5_to_dictionary(ground_truth)

    keys_truth = list(truth.keys())
    for key in generated.keys():
        assert key in keys_truth, f"generated key: {key} does not exist in saved benchmark s500 sonar file"
        assert (truth[key] == generated[key]).all()
    os.remove(read_out)
