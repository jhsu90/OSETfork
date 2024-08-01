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

def peak_det_likelihood_unit_test():
    data = scipy.io.loadmat('/Users/jasper/Desktop/My_Life/GaTech/CliffordLab/codes/OSETfork/datasets/sample-data/peak.mat')
    peak_matlab = data['peak'][0]
    peak_python, _, _, _ = peak_det_likelihood(np.array(mat), fs)
    i = 11
    return testing.compare_number_arrays(peak_python, peak_matlab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This is a unit test for peak_det_likelihood"""
    )
    args = parser.parse_args()
    print(peak_det_likelihood_unit_test())


