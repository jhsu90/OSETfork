import pywt
from .peak_det_likelihood_helper import *
from oset.generic.tanh_saturation import tanh_saturation

# Your code using the oset package

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
        # left_pad_len = left_pad.shape[1]

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
    # works only for fileter_type: WAVELET
    if 'filter_type' not in params or params['filter_type'] is None:
        params['filter_type'] = 'WAVELET'
        if params["verbose"]:
            print(f"params.filter_type = {params['filter_type']}")

    if params['filter_type'] == 'WAVELET':
        # Set wavelet type
        if 'wden_type' not in params or params['wden_type'] is None:
            params['wden_type'] = 'sym4'
            if params["verbose"]:
                print(f"params.wden_type = {params['wden_type']}")

        # Set upper level for denoising
        if 'wden_upper_level' not in params or params['wden_upper_level'] is None:
            f_high = min(49.0, fs / 2)
            params['wden_upper_level'] = round(np.log2(fs / f_high))
            if params["verbose"]:
                print(f"params.wden_upper_level = {params['wden_upper_level']}")

        # Set lower level for denoising
        if 'wden_lower_level' not in params or params['wden_lower_level'] is None:
            f_low = 18.0
            params['wden_lower_level'] = round(np.log2(fs / f_low))
            if params["verbose"]:
                print(f"params.wden_lower_level = {params['wden_lower_level']}")

        if params['wden_upper_level'] > params['wden_lower_level']:
            raise ValueError('Requested wavelet band too narrow, or sampling frequency is too low.')

        data_filtered_padded = np.zeros_like(data_padded)
        for kk in range(data_padded.shape[0]):
            wt = pywt.swt(data_padded[kk, :], params['wden_type'], level=params['wden_lower_level']) # originally modwt
            wtrec = np.zeros_like(wt)
            wtrec[params['wden_upper_level']:params['wden_lower_level'] + 1, :] = wt[params['wden_upper_level']:params['wden_lower_level'] + 1, :]
            data_filtered_padded[kk, :] = pywt.iswt(wtrec, params['wden_type']) # originally imodwt

            if params["PLOT_DIAGNOSTIC"]:
                plt.figure()
                plt.plot(data_padded[kk, :], label='data')
                plt.plot(data_padded[kk, :] - data_filtered_padded[kk, :], label='baseline')
                plt.plot(data_filtered_padded[kk, :], label='data_filtered')
                plt.grid(True)
                plt.legend()
                plt.title('Wavelet-based preprocessing filter')
                plt.show()

    else:
        raise ValueError('Unknown preprocessing filter')

    # Calculate the residual signal (after filtering)
    data_residual_padded = data_padded - data_filtered_padded
    if 'f_baseline_cutoff' not in params or params['f_baseline_cutoff'] is None:
        params['f_baseline_cutoff'] = 0.5
        if params["verbose"]:
            print(f"params.f_baseline_cutoff = {params['f_baseline_cutoff']}")
    data_residual_padded -= lp_filter_zero_phase(data_residual_padded, params['f_baseline_cutoff'] / fs)

    # saturate the channels at k_sigma times the STD of each channel
    if 'saturate' not in params or params['saturate'] is None:
        params['saturate'] = 1
        if params["verbose"]:
            print(f"params.saturate = {params['saturate']}")

    if params['saturate'] == 1 or params['sat_k_sigma'] is None:
        if 'sat_k_sigma' not in params:
            params['sat_k_sigma'] = 8.0
            if params["verbose"]:
                print(f"params.sat_k_sigma = {params['sat_k_sigma']}")
        data_filtered_padded = tanh_saturation(data_filtered_padded, params['sat_k_sigma'])
        data_residual_padded = tanh_saturation(data_residual_padded, params['sat_k_sigma'])
