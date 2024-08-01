import argparse

import matlab
import matlab.engine
import scipy.io
import numpy as np

import unit_test as testing

from oset.ecg.peak_detection.peak_det_likelihood import peak_det_likelihood

mat = scipy.io.loadmat("../../../datasets/sample-data/SampleECGData.mat")["ECGdata"]
fs = 1000
mat = mat[:, 0: 60 * fs]

def peak_det_likelihood_unit_test_multiple_channels():
    ml = runMatLab()
    py = runPython()
    return testing.compare_number_arrays(
        py[1] + 1,
        np.array(ml[1][0]).astype(int)[0]
    )

def runMatLab():
    eng = matlab.engine.start_matlab()
    x = matlab.double(mat.tolist())
    params = {}
    params['filter_type'] = 'BANDPASS_FILTER'
    eng.addpath("../../../matlab/tools/ecg")
    eng.addpath("../../../matlab/tools/generic")
    return eng.peak_det_likelihood(x, float(fs), params, nargout=2)

def runPython():
    return peak_det_likelihood(mat, fs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This is a unit test for peak_det_likelihood"""
    )
    args = parser.parse_args()
    print(peak_det_likelihood_unit_test_multiple_channels())


