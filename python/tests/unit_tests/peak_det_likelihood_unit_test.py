import argparse

import matlab
import matlab.engine
import scipy.io
import numpy as np

import unit_test as testing

from oset.ecg.peak_detection.peak_det_likelihood import peak_det_likelihood

mat = scipy.io.loadmat("../../../datasets/sample-data/SampleECGData.mat")["ECGdata"]
fs = 1000


def peak_det_likelihood_unit_test():
    data = scipy.io.loadmat('/Users/jasper/Desktop/My_Life/GaTech/CliffordLab/codes/OSETfork/datasets/sample-data/data_filtered_env1.mat')
    double_value_matlab = data['data_filtered_env1'][0]
    double_value_python, _ = peak_det_likelihood(np.array(mat), fs)
    return testing.compare_number_arrays(double_value_python, double_value_matlab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This is a unit test for peak_det_likelihood"""
    )
    args = parser.parse_args()
    print(peak_det_likelihood_unit_test())


