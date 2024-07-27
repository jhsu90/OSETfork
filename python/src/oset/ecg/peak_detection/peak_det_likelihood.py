import pywt
from .peak_det_likelihood_helper import *
from oset.generic.tanh_saturation import tanh_saturation
from oset.generic.lp_filter.lp_filter_zero_phase import lp_filter_zero_phase
from scipy.signal import filtfilt, medfilt, butter

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
    # Default filter type: BANDPASS_FILTER
    if 'filter_type' not in params or params['filter_type'] is None:
        params['filter_type'] = 'BANDPASS_FILTER'
        if params["verbose"]:
            print(f"params.filter_type = {params['filter_type']}")

    # if params['filter_type'] == 'WAVELET':
    #     # Set wavelet type
    #     if 'wden_type' not in params or params['wden_type'] is None:
    #         params['wden_type'] = 'sym4'
    #         if params["verbose"]:
    #             print(f"params.wden_type = {params['wden_type']}")
    #
    #     # Set upper level for denoising
    #     if 'wden_upper_level' not in params or params['wden_upper_level'] is None:
    #         f_high = min(49.0, fs / 2)
    #         params['wden_upper_level'] = round(np.log2(fs / f_high))
    #         if params["verbose"]:
    #             print(f"params.wden_upper_level = {params['wden_upper_level']}")
    #
    #     # Set lower level for denoising
    #     if 'wden_lower_level' not in params or params['wden_lower_level'] is None:
    #         f_low = 18.0
    #         params['wden_lower_level'] = round(np.log2(fs / f_low))
    #         if params["verbose"]:
    #             print(f"params.wden_lower_level = {params['wden_lower_level']}")
    #
    #     if params['wden_upper_level'] > params['wden_lower_level']:
    #         raise ValueError('Requested wavelet band too narrow, or sampling frequency is too low.')
    #
    #     data_filtered_padded = np.zeros_like(data_padded)
    #     for kk in range(data_padded.shape[0]):
    #         wt = pywt.swt(data_padded[kk, :], params['wden_type'], level=params['wden_lower_level']) # originally modwt
    #         wtrec = np.zeros_like(wt)
    #         wtrec[params['wden_upper_level']:params['wden_lower_level'] + 1, :] = wt[params['wden_upper_level']:params['wden_lower_level'] + 1, :]
    #         data_filtered_padded[kk, :] = pywt.iswt(wtrec, params['wden_type']) # originally imodwt
    #
    #         if params["PLOT_DIAGNOSTIC"]:
    #             plt.figure()
    #             plt.plot(data_padded[kk, :], label='data')
    #             plt.plot(data_padded[kk, :] - data_filtered_padded[kk, :], label='baseline')
    #             plt.plot(data_filtered_padded[kk, :], label='data_filtered')
    #             plt.grid(True)
    #             plt.legend()
    #             plt.title('Wavelet-based preprocessing filter')
    #             plt.show()

    if params['filter_type'] == 'BANDPASS_FILTER':
        if 'bp_lower_cutoff' not in params or params['bp_lower_cutoff'] is None:
            params['bp_lower_cutoff'] = 8.0
            if params.get('verbose', False):
                print(f"params.bp_lower_cutoff = {params['bp_lower_cutoff']}")

        if 'bp_upper_cutoff' not in params or params['bp_upper_cutoff'] is None:
            params['bp_upper_cutoff'] = 40.0
            if params.get('verbose', False):
                print(f"params.bp_upper_cutoff = {params['bp_upper_cutoff']}")

        data_lp_upper = lp_filter_zero_phase(data_padded, params['bp_upper_cutoff'] / fs)
        data_lp_lower = lp_filter_zero_phase(data_padded, params['bp_lower_cutoff'] / fs)
        data_filtered_padded = data_lp_upper - data_lp_lower
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

    # calculate the power envelope of one or all channels (stage 1)
    if 'power_env_wlen' not in params or params['power_env_wlen'] is None:
        params['power_env_wlen'] = 0.025
        if params["verbose"]:
            print(f"params.power_env_wlen = {params['power_env_wlen']}")

    power_env_wlen = int(np.ceil(params['power_env_wlen'] * fs))

    data_filtered = data_filtered_padded[:, left_pad_len : left_pad_len + sig_len]

    # signal power
    filter_kernel = np.ones(power_env_wlen) / power_env_wlen
    data_filtered_env1_padded = filtfilt(filter_kernel, 1, np.sqrt(np.mean(data_filtered_padded**2, axis=0)))
    data_filtered_env1 = data_filtered_env1_padded[left_pad_len: left_pad_len + sig_len]

    # residual power
    data_residual = data_residual_padded[:, left_pad_len + 1 : left_pad_len + sig_len]
    data_residual_env1_padded = filtfilt(filter_kernel, 1, np.sqrt(np.mean(data_residual_padded**2, axis=0)))
    # data_residual_env1 = data_residual_env1_padded[left_pad_len: left_pad_len + sig_len]

    # calculate the power envelope of one or all channels (stage 2)
    if 'two_stage_env' not in params or params['two_stage_env'] is None:
        params['two_stage_env'] = True
        if params["verbose"]:
            print(f"params.two_stage_env = {params['two_stage_env']}")

    if params['two_stage_env'] == 1:
        if 'power_env_wlen2' not in params or params['power_env_wlen2'] is None:
            params['power_env_wlen2'] = 0.075
            if params["verbose"]:
                print(f"params.power_env_wlen2 = {params['power_env_wlen2']}")

        power_env_wlen2 = int(np.ceil(params['power_env_wlen2'] * fs))
        filter_kernel = np.ones(power_env_wlen2) / power_env_wlen2

        # signal power
        data_filtered_env2_padded = filtfilt(filter_kernel, 1, np.sqrt(np.mean(data_filtered_padded ** 2, axis=0)))
        data_filtered_env2 = data_filtered_env2_padded[left_pad_len + 1: left_pad_len + sig_len]

        # residual power
        data_residual_env2_padded = filtfilt(filter_kernel, 1, np.sqrt(np.mean(data_residual_padded ** 2, axis=0)))
        # data_residual_env2 = data_residual_env2_padded[left_pad_len + 1: left_pad_len + sig_len]

        # signal power combined (two stages)
        data_filtered_env_padded = np.sqrt(np.abs(data_filtered_env1_padded * data_filtered_env2_padded))
        data_filtered_env = np.sqrt(np.abs(data_filtered_env1 * data_filtered_env2))

        # residual power combined (two stages)
        data_residual_env_padded = np.sqrt(np.abs(data_residual_env1_padded * data_residual_env2_padded))
        # data_residual_env = np.sqrt(np.abs(data_residual_env1 * data_residual_env2))

        if params.get('PLOT_DIAGNOSTIC', False):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(data_filtered_padded, label='Filtered Data')
            plt.plot(data_filtered_env1_padded, label='First Stage Envelope')
            plt.plot(data_filtered_env2, label='Second Stage Envelope')
            plt.plot(data_filtered_env, label='Combined Envelope')
            plt.grid(True)
            plt.legend()
            plt.title('Signal Envelope Estimates')
            plt.show()

    else:
        data_filtered_env = data_filtered_env1
        data_filtered_env_padded = data_filtered_env1_padded
        # data_residual_env = data_residual_env1
        data_residual_env_padded = data_residual_env1_padded

    data_mn_all_channels_padded = np.mean(data_padded, axis=0)
    data_filtered_mn_all_channels_padded = np.mean(data_filtered_padded, axis=0)
