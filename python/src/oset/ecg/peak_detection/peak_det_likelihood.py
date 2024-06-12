import numpy as np


def peak_det_likelihood(data, fs, params=None):
    # Default parameter setup
    if params is None:
        params = {}
    params.setdefault("verbose", False)
    params.setdefault("PLOT_DIAGNOSTIC", False)
    params.setdefault("saturate", True)
    # params.setdefault("filter_type", "WAVELET")
    # params.setdefault('bp_lower_cutoff', 1) # 8
    # params.setdefault('bp_upper_cutoff', 40)
    # params.setdefault('gaus_match_filt_span', 0.2) # 0.1
    # params.setdefault('gaus_match_filt_sigma', 0.01)
    # params.setdefault('power_env_wlen', 0.03) # 0.025
    # params.setdefault('n_bins', max(250, min(500, int(0.1 * data.shape[1]))))
    # params.setdefault('power_env_hist_peak_th', 0.9)
    # params.setdefault('min_peak_distance', 0.2)
    # params.setdefault('likelihood_sigma', 0.01)
    # params.setdefault('max_likelihood_span', 0.1)
    # params.setdefault('RemoveUnsimilarBeats', True)

    if params["verbose"]:
        print("Operating in verbose mode. Default settings will be displayed.")

    if params["PLOT_DIAGNOSTIC"]:
        print(
            "Operating in diagnostic plot mode. All peak refinement figures will be plotted."
        )

    sig_len = data.shape[1]  # signal length

    # Check if the signal is longer than 60 seconds
    if sig_len / fs > 60.0:
        print(
            f"The signal is {sig_len / fs:.1f}s long; consider using a function for long recordings."
        )

    # left and right padding for signal continuity
    if "left_pad_len" not in params or params["left_pad_len"] is None:
        left_pad_len = int(round(1.0 * fs))
        if params["verbose"]:
            print(f"left_pad_len = {left_pad_len}")
    else:
        left_pad_len = params["left_pad_len"]

    if "right_pad_len" not in params or params["right_pad_len"] is None:
        right_pad_len = int(round(1.0 * fs))
        if params["verbose"]:
            print(f"right_pad_len = {right_pad_len}")
    else:
        right_pad_len = params["right_pad_len"]

    if "left_pad" not in params or params["left_pad"] is None:
        left_pad = np.zeros((data.shape[0], left_pad_len))
    else:
        if data.shape[0] != params["left_pad"].shape[0]:
            raise ValueError("size(data, 0) != size(params['left_pad'], 0)")
        left_pad = params["left_pad"]
        left_pad_len = left_pad.shape[1]

    if "right_pad" not in params or params["right_pad"] is None:
        right_pad = np.zeros((data.shape[0], right_pad_len))
    else:
        if data.shape[0] != params["right_pad"].shape[0]:
            raise ValueError("size(data, 0) != size(params['right_pad'], 0)")
        right_pad = params["right_pad"]
        # right_pad_len = right_pad.shape[1]

    # Concatenate left_pad, data, and right_pad
    data_padded = np.hstack((left_pad, data, right_pad))

    # pass the channels through a narrow bandpass filter or a matched filter to remove the baseline and the T-waves


