import pywt
from .peak_det_likelihood_helper import *
from oset.generic.tanh_saturation import tanh_saturation
from oset.generic.lp_filter.lp_filter_zero_phase import lp_filter_zero_phase
from scipy.signal import filtfilt, medfilt, butter


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

    data_padded = np.hstack((left_pad, data, right_pad))

    """
    pass the channels through a narrow bandpass filter or a matched filter to remove the baseline and the T-waves
    """
    # Default filter type: BANDPASS_FILTER
    # Other filter types not yet implemented
    if "filter_type" not in params or params["filter_type"] is None:
        params["filter_type"] = "BANDPASS_FILTER"
        if params["verbose"]:
            print(f"params.filter_type = {params['filter_type']}")

    if params["filter_type"] == "BANDPASS_FILTER":
        if "bp_lower_cutoff" not in params or params["bp_lower_cutoff"] is None:
            params["bp_lower_cutoff"] = 8.0
            if params.get("verbose", False):
                print(f"params.bp_lower_cutoff = {params['bp_lower_cutoff']}")

        if "bp_upper_cutoff" not in params or params["bp_upper_cutoff"] is None:
            params["bp_upper_cutoff"] = 40.0
            if params.get("verbose", False):
                print(f"params.bp_upper_cutoff = {params['bp_upper_cutoff']}")

        data_lp_upper = lp_filter_zero_phase(
            data_padded, params["bp_upper_cutoff"] / fs
        )
        data_lp_lower = lp_filter_zero_phase(
            data_padded, params["bp_lower_cutoff"] / fs
        )
        data_filtered_padded = data_lp_upper - data_lp_lower
    else:
        raise ValueError("Unknown preprocessing filter")

    """
    Calculate the residual signal (after filtering)
    """
    data_residual_padded = data_padded - data_filtered_padded
    if "f_baseline_cutoff" not in params or params["f_baseline_cutoff"] is None:
        params["f_baseline_cutoff"] = 0.5
        if params["verbose"]:
            print(f"params.f_baseline_cutoff = {params['f_baseline_cutoff']}")
    data_residual_padded -= lp_filter_zero_phase(
        data_residual_padded, params["f_baseline_cutoff"] / fs
    )

    """
    saturate the channels at k_sigma times the STD of each channel
    """
    if "saturate" not in params or params["saturate"] is None:
        params["saturate"] = 1
        if params["verbose"]:
            print(f"params.saturate = {params['saturate']}")

    if params["saturate"] == 1 or params["sat_k_sigma"] is None:
        if "sat_k_sigma" not in params:
            params["sat_k_sigma"] = 8.0
            if params["verbose"]:
                print(f"params.sat_k_sigma = {params['sat_k_sigma']}")
        data_filtered_padded = tanh_saturation(
            data_filtered_padded, params["sat_k_sigma"]
        )
        data_residual_padded = tanh_saturation(
            data_residual_padded, params["sat_k_sigma"]
        )

    """
    calculate the power envelope of one or all channels (stage 1)
    """
    if "power_env_wlen" not in params or params["power_env_wlen"] is None:
        params["power_env_wlen"] = 0.025
        if params["verbose"]:
            print(f"params.power_env_wlen = {params['power_env_wlen']}")

    power_env_wlen = int(np.ceil(params["power_env_wlen"] * fs))

    data_filtered = data_filtered_padded[:, left_pad_len : left_pad_len + sig_len]

    # signal power
    filter_kernel = np.ones(power_env_wlen) / power_env_wlen
    data_filtered_env1_padded = filtfilt(
        filter_kernel, 1, np.sqrt(np.mean(data_filtered_padded**2, axis=0))
    )
    data_filtered_env1 = data_filtered_env1_padded[
        left_pad_len : left_pad_len + sig_len
    ]

    # residual power
    data_residual = data_residual_padded[:, left_pad_len + 1 : left_pad_len + sig_len]
    data_residual_env1_padded = filtfilt(
        filter_kernel, 1, np.sqrt(np.mean(data_residual_padded**2, axis=0))
    )
    # data_residual_env1 = data_residual_env1_padded[left_pad_len: left_pad_len + sig_len]

    """
    calculate the power envelope of one or all channels (stage 2)
    """
    if "two_stage_env" not in params or params["two_stage_env"] is None:
        params["two_stage_env"] = True
        if params["verbose"]:
            print(f"params.two_stage_env = {params['two_stage_env']}")

    if params["two_stage_env"] == 1:
        if "power_env_wlen2" not in params or params["power_env_wlen2"] is None:
            params["power_env_wlen2"] = 0.075
            if params["verbose"]:
                print(f"params.power_env_wlen2 = {params['power_env_wlen2']}")

        power_env_wlen2 = int(np.ceil(params["power_env_wlen2"] * fs))
        filter_kernel = np.ones(power_env_wlen2) / power_env_wlen2

        # signal power
        data_filtered_env2_padded = filtfilt(
            filter_kernel, 1, np.sqrt(np.mean(data_filtered_padded**2, axis=0))
        )
        data_filtered_env2 = data_filtered_env2_padded[
            left_pad_len : left_pad_len + sig_len
        ]

        # residual power
        data_residual_env2_padded = filtfilt(
            filter_kernel, 1, np.sqrt(np.mean(data_residual_padded**2, axis=0))
        )
        # data_residual_env2 = data_residual_env2_padded[left_pad_len + 1: left_pad_len + sig_len]

        # signal power combined (two stages)
        data_filtered_env_padded = np.sqrt(
            np.abs(data_filtered_env1_padded * data_filtered_env2_padded)
        )
        data_filtered_env = np.sqrt(np.abs(data_filtered_env1 * data_filtered_env2))

        # residual power combined (two stages)
        data_residual_env_padded = np.sqrt(
            np.abs(data_residual_env1_padded * data_residual_env2_padded)
        )
        # data_residual_env = np.sqrt(np.abs(data_residual_env1 * data_residual_env2))

        if params["PLOT_DIAGNOSTIC"]:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(data_filtered_padded, label="Filtered Data")
            plt.plot(data_filtered_env1_padded, label="First Stage Envelope")
            plt.plot(data_filtered_env2, label="Second Stage Envelope")
            plt.plot(data_filtered_env, label="Combined Envelope")
            plt.grid(True)
            plt.legend()
            plt.title("Signal Envelope Estimates")
            plt.show()

    else:
        data_filtered_env = data_filtered_env1
        data_filtered_env_padded = data_filtered_env1_padded
        # data_residual_env = data_residual_env1
        data_residual_env_padded = data_residual_env1_padded

    data_mn_all_channels_padded = np.mean(data_padded, axis=0)
    data_filtered_mn_all_channels_padded = np.mean(data_filtered_padded, axis=0)

    """
    find the envelope bumps (the top percentile of the signal's power envelope and above the noise level)
    for R-peak search
    """
    if (
        "power_env_hist_peak_th" not in params
        or params["power_env_hist_peak_th"] is None
    ):
        if "p_residual_th" not in params or params["p_residual_th"] is None:
            params["p_residual_th"] = 95.0
        p_residual = np.percentile(data_residual_env_padded, params["p_residual_th"])

        if "p_signal_th" not in params or params["p_signal_th"] is None:
            params["p_signal_th"] = 85.0
        p_signal = np.percentile(data_filtered_env_padded, params["p_signal_th"])

        if p_signal > p_residual:
            params["power_env_hist_peak_th"] = (p_residual + p_signal) / 2
        else:
            params["power_env_hist_peak_th"] = p_signal

        if params["verbose"]:
            print(f"params.power_env_hist_peak_th = {params['power_env_hist_peak_th']}")
            print(f"params.p_residual_th = {params['p_residual_th']}")
            print(f"params.p_signal_th = {params['p_signal_th']}")
            print(f"p_residual = {p_residual}")
            print(f"p_signal = {p_signal}")

    bumps_indexes_padded, _ = refine_peaks_low_amp_peaks_prctile(
        data_filtered_env_padded,
        range(1, len(data_filtered_env_padded)),
        "LEVEL",
        params["power_env_hist_peak_th"],
        params["PLOT_DIAGNOSTIC"],
    )
    bumps_indexes = bumps_indexes_padded - left_pad_len
    bumps_indexes = bumps_indexes[(bumps_indexes >= 0) & (bumps_indexes < sig_len)]

    """
    search for all local peaks within a given minimal sliding window length
    """
    if "min_peak_distance" not in params or params["min_peak_distance"] is None:
        params["min_peak_distance"] = 0.2
        if params["verbose"]:
            print(f"params.min_peak_distance = {params['min_peak_distance']}")

    rpeak_search_half_wlen = int(np.floor(fs * params["min_peak_distance"]))

    env_pk_detect_mode = "POS"
    if params["verbose"]:
        print(f"params.env_pk_detect_mode = {params['env_pk_detect_mode']}")
    peak_indexes_padded, _ = refine_peaks_too_close_low_amp(
        data_filtered_env_padded,
        bumps_indexes_padded,
        rpeak_search_half_wlen,
        env_pk_detect_mode,
        params["PLOT_DIAGNOSTIC"],
    )
    peak_indexes = peak_indexes_padded - left_pad_len
    peak_indexes = peak_indexes[(peak_indexes >= 0) & (peak_indexes < sig_len)]

    # # first or last samples are not selected as peaks
    # if len(peak_indexes) > 0 and peak_indexes[0] == 0:
    #     peak_indexes = peak_indexes[1:]
    # if len(peak_indexes) > 0 and peak_indexes[-1] == sig_len - 1:
    #     peak_indexes = peak_indexes[:-1]

    """
    matched filter using average beat shape
    """
    if (
        "ENHANCE_MATCHED_FILTER" not in params
        or params["ENHANCE_MATCHED_FILTER"] is None
    ):
        params["ENHANCE_MATCHED_FILTER"] = True
        if params["verbose"]:
            print(f"params.ENHANCE_MATCHED_FILTER = {params['ENHANCE_MATCHED_FILTER']}")

    if params["ENHANCE_MATCHED_FILTER"]:
        data_filtered_enhanced_padded, data_filtered_enhanced_env_padded = (
            signal_specific_matched_filter(data_filtered_padded, peak_indexes_padded)
        )

        data_filtered_enhanced = data_filtered_enhanced_padded[
            :, left_pad_len : left_pad_len + sig_len
        ]
        data_filtered_enhanced_env = data_filtered_enhanced_env_padded[
            left_pad_len : left_pad_len + sig_len
        ]  # modified to comply with python style slicing

        if params["PLOT_DIAGNOSTIC"]:
            plt.figure()
            plt.plot(data_filtered_padded.mean(axis=0), label="Original Filtered Data")
            plt.plot(
                data_filtered_enhanced.mean(axis=0), label="Enhanced Filtered Data"
            )
            plt.plot(
                data_filtered_enhanced_env.mean(axis=0),
                label="Enhanced Filtered Envelope",
            )
            plt.title("Matched Filter-Based Signal and Envelope Enhancement")
            plt.legend()
            plt.grid(True)
            plt.show()
        data_filtered = data_filtered_enhanced
        data_filtered_env = data_filtered_enhanced_env
        data_filtered_padded = data_filtered_enhanced_padded
        data_filtered_env_padded = data_filtered_enhanced_env_padded
        data_mn_all_channels_padded = np.mean(data_padded, 0)
        data_filtered_mn_all_channels_padded = np.mean(data_filtered_padded, 0)

    """
    Refine the extracted R-peaks
    """

    peaks_all_method = np.zeros(sig_len)
    peaks_all_method[peak_indexes] = 1

    if "weight_base_method" not in params or params["weight_base_method"] is None:
        params["weight_base_method"] = 2.0
        if params["verbose"]:
            print(f"params.weight_base_method = {params['weight_base_method']}")

    consensus_weights = [params["weight_base_method"]]
    refinement_methods = ["PRE_REFINEMENT"]

    if "REFINE_PEAKS" not in params or params["REFINE_PEAKS"] is None:
        params["REFINE_PEAKS"] = True
        if params["verbose"]:
            print(f"params.REFINE_PEAKS = {params['REFINE_PEAKS']}")

    if params["REFINE_PEAKS"] and len(peak_indexes) > 1:

        """
        detect beats that were most impacted by preprocessing filter (for T-wave detection purposes)
        (a lot different from matlab result)
        """
        if (
            "OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC" not in params
            or params["OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC"] is None
        ):
            params["OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC"] = True
            if params["verbose"]:
                print(
                    f"params.OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC = {params['OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC']}"
                )

            pparams = {}
            pparams["ff_low_cutoff"] = 3.0 / fs
            pparams["ff_high_cutoff"] = min(0.5, 50.0 / fs)
            pparams["pre_to_post_filter_power_ratio_th"] = 3.0
            pparams["percentile"] = 90.0
            pparams["percentile_fraction"] = 0.25

            if params["verbose"]:
                print(f"pparams.ff_low_cutoff = {pparams['ff_low_cutoff']}")
                print(f"pparams.ff_high_cutoff = {pparams['ff_high_cutoff']}")
                print(
                    f"pparams.pre_to_post_filter_power_ratio_th = {pparams['pre_to_post_filter_power_ratio_th']}"
                )
                print(f"pparams.percentile = {pparams['percentile']}")
                print(f"pparams.percentile_fraction = {pparams['percentile_fraction']}")

        if params["OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC"]:
            _, peaks_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC = (
                refine_peaks_filter_energy_impacted_beats(
                    data_mn_all_channels_padded,
                    data_filtered_mn_all_channels_padded,
                    peak_indexes_padded,
                    pparams,
                )
            )
            peaks_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC = (
                peaks_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC[
                    left_pad_len : left_pad_len + sig_len
                ]
            )

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC)
            )
            refinement_methods.append("OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC")

            if "weight_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC" not in params:
                params["weight_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC = {params['weight_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC']}"
                    )

            consensus_weights.append(
                params["weight_OMIT_BEATS_MOST_IMPACTED_BY_PRE_PROC"]
            )

        """
        detect beats with extreme amplitude deviations from other peaks
        (one peak off from matlab result)
        """
        if (
            "OMIT_LOW_AMP_PEAKS_PRCTL_FRAC" not in params
            or params["OMIT_LOW_AMP_PEAKS_PRCTL_FRAC"] is None
        ):
            params["OMIT_LOW_AMP_PEAKS_PRCTL_FRAC"] = True
            if params.get("verbose", False):
                print(
                    f"params.OMIT_LOW_AMP_PEAKS_PRCTL_FRAC = {params['OMIT_LOW_AMP_PEAKS_PRCTL_FRAC']}"
                )

        if params["OMIT_LOW_AMP_PEAKS_PRCTL_FRAC"]:
            pparams = {}
            pparams["percentile"] = 90.0
            pparams["percentile_fraction"] = 0.3
            if params.get("verbose", False):
                print(f"pparams.percentile = {pparams['percentile']}")
                print(f"pparams.percentile_fraction = {pparams['percentile_fraction']}")

            _, peaks_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC = (
                refine_peaks_low_amp_peaks_prctile_fraction(
                    data_filtered_env,
                    peak_indexes,
                    pparams,
                    params.get("PLOT_DIAGNOSTIC", False),
                )
            )
            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC)
            )
            refinement_methods.append("OMIT_LOW_AMP_PEAKS_PRCTL_FRAC")

            if "weight_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC" not in params:
                params["weight_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC = {params['weight_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC']}"
                    )

            consensus_weights.append(params["weight_OMIT_LOW_AMP_PEAKS_PRCTL_FRAC"])
        """
        detect beats with low variance
        (same with matlab result)
        """
        if (
            "OMIT_LOW_POWER_BEATS" not in params
            or params["OMIT_LOW_POWER_BEATS"] is None
        ):
            params["OMIT_LOW_POWER_BEATS"] = True
            pparams = {}
            pparams["beat_std_med_frac_th"] = 0.5
            pparams["max_amp_prctile"] = 90.0
            if params.get("verbose", False):
                print(f"params.OMIT_LOW_POWER_BEATS = {params['OMIT_LOW_POWER_BEATS']}")
                print(
                    f"   pparams.beat_std_med_frac_th = {params['beat_std_med_frac_th']}"
                )
                print(f"   pparams.max_amp_prctile = {params['max_amp_prctile']}")

        if params["OMIT_LOW_POWER_BEATS"]:
            _, peaks_OMIT_LOW_POWER_BEATS = refine_peaks_low_power_beats(
                data_filtered_mn_all_channels_padded,
                peak_indexes_padded,
                pparams["max_amp_prctile"],
                pparams["beat_std_med_frac_th"],
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_LOW_POWER_BEATS = peaks_OMIT_LOW_POWER_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack((peaks_all_method, peaks_OMIT_LOW_POWER_BEATS))
            refinement_methods.append("OMIT_LOW_POWER_BEATS")

            if (
                "weight_OMIT_LOW_POWER_BEATS" not in params
                or params["weight_OMIT_LOW_POWER_BEATS"] is None
            ):
                params["weight_OMIT_LOW_POWER_BEATS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_LOW_POWER_BEATS = {params['weight_OMIT_LOW_POWER_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_LOW_POWER_BEATS"])

        """
        detect beats with negative correlations with the average beat
        (same)
        """
        if (
            "OMIT_NEG_CORRCOEF_BEATS" not in params
            or params["OMIT_NEG_CORRCOEF_BEATS"] is None
        ):
            params["OMIT_NEG_CORRCOEF_BEATS"] = True
            if params.get("verbose", False):
                print(
                    f"params.OMIT_NEG_CORRCOEF_BEATS = {params['OMIT_NEG_CORRCOEF_BEATS']}"
                )

        if params["OMIT_NEG_CORRCOEF_BEATS"]:
            _, peaks_OMIT_NEG_CORRCOEF_BEATS = refine_peaks_waveform_similarity(
                data_filtered_mn_all_channels_padded,
                peak_indexes_padded,
                [],
                "NEG-CORR",
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_NEG_CORRCOEF_BEATS = peaks_OMIT_NEG_CORRCOEF_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_NEG_CORRCOEF_BEATS)
            )
            refinement_methods.append("OMIT_NEG_CORRCOEF_BEATS")

            if (
                "weight_OMIT_NEG_CORRCOEF_BEATS" not in params
                or params["weight_OMIT_NEG_CORRCOEF_BEATS"] is None
            ):
                params["weight_OMIT_NEG_CORRCOEF_BEATS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_NEG_CORRCOEF_BEATS = {params['weight_OMIT_NEG_CORRCOEF_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_NEG_CORRCOEF_BEATS"])

        """
        detect beats with low correlation coefficient with the average beat
        (same)
        """
        if (
            "OMIT_LOW_CORRCOEF_BEATS" not in params
            or params["OMIT_LOW_CORRCOEF_BEATS"] is None
        ):
            params["OMIT_LOW_CORRCOEF_BEATS"] = True
            pparams = {}
            pparams["k_sigma"] = 3.0
            if params.get("verbose", False):
                print(
                    f"params.OMIT_LOW_CORRCOEF_BEATS = {params['OMIT_LOW_CORRCOEF_BEATS']}"
                )
                print(f"   pparams.k_sigma = {params['k_sigma']}")

        if params["OMIT_LOW_CORRCOEF_BEATS"]:
            _, peaks_OMIT_LOW_CORRCOEF_BEATS = refine_peaks_waveform_similarity(
                data_filtered_env_padded,
                peak_indexes_padded,
                pparams,
                "BEAT-STD",
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_LOW_CORRCOEF_BEATS = peaks_OMIT_LOW_CORRCOEF_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_LOW_CORRCOEF_BEATS)
            )
            refinement_methods.append("OMIT_LOW_CORRCOEF_BEATS")

            if (
                "weight_OMIT_LOW_CORRCOEF_BEATS" not in params
                or params["weight_OMIT_LOW_CORRCOEF_BEATS"] is None
            ):
                params["weight_OMIT_LOW_CORRCOEF_BEATS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_LOW_CORRCOEF_BEATS = {params['weight_OMIT_LOW_CORRCOEF_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_LOW_CORRCOEF_BEATS"])

        """
        detect beats with low correlation with other beats
        (same)
        """
        if "OMIT_LOW_CORR_BEATS" not in params or params["OMIT_LOW_CORR_BEATS"] is None:
            params["OMIT_LOW_CORR_BEATS"] = True
            pparams = {}
            pparams["percentile"] = 90.0
            pparams["percentile_fraction"] = 0.3
            if params.get("verbose", False):
                print(f"params.OMIT_LOW_CORR_BEATS = {params['OMIT_LOW_CORR_BEATS']}")
                print(f"   pparams.percentile = {params['percentile']}")
                print(
                    f"   pparams.percentile_fraction = {params['percentile_fraction']}"
                )

        if params["OMIT_LOW_CORR_BEATS"]:
            _, peaks_OMIT_LOW_CORR_BEATS = refine_peaks_waveform_similarity(
                data_filtered_mn_all_channels_padded,
                peak_indexes_padded,
                pparams,
                "CORR",
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_LOW_CORR_BEATS = peaks_OMIT_LOW_CORR_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack((peaks_all_method, peaks_OMIT_LOW_CORR_BEATS))
            refinement_methods.append("OMIT_LOW_CORR_BEATS")

            if (
                "weight_OMIT_LOW_CORR_BEATS" not in params
                or params["weight_OMIT_LOW_CORR_BEATS"] is None
            ):
                params["weight_OMIT_LOW_CORR_BEATS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_LOW_CORR_BEATS = {params['weight_OMIT_LOW_CORR_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_LOW_CORR_BEATS"])
        """
        detect beats with low correlation coefficient with other beats
        (same)
        """
        if (
            "OMIT_LOW_CORRCOEF_BEATS" not in params
            or params["OMIT_LOW_CORRCOEF_BEATS"] is None
        ):
            params["OMIT_LOW_CORRCOEF_BEATS"] = False
            pparams = {}
            pparams["percentile"] = 90.0
            pparams["percentile_fraction"] = 0.3
            if params.get("verbose", False):
                print(
                    f"params.OMIT_LOW_CORRCOEF_BEATS = {params['OMIT_LOW_CORRCOEF_BEATS']}"
                )
                print(f"   pparams.percentile = {params['percentile']}")
                print(
                    f"   pparams.percentile_fraction = {params['percentile_fraction']}"
                )

        if params["OMIT_LOW_CORRCOEF_BEATS"]:
            _, peaks_OMIT_LOW_CORRCOEF_BEATS = refine_peaks_waveform_similarity(
                data_filtered_mn_all_channels_padded,
                peak_indexes_padded,
                pparams,
                "CORRCOEF",
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_LOW_CORRCOEF_BEATS = peaks_OMIT_LOW_CORRCOEF_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_LOW_CORRCOEF_BEATS)
            )
            refinement_methods.append("OMIT_LOW_CORRCOEF_BEATS")

            if (
                "weight_OMIT_LOW_CORRCOEF_BEATS" not in params
                or params["weight_OMIT_LOW_CORRCOEF_BEATS"] is None
            ):
                params["weight_OMIT_LOW_CORRCOEF_BEATS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_LOW_CORRCOEF_BEATS = {params['weight_OMIT_LOW_CORRCOEF_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_LOW_CORRCOEF_BEATS"])

        """
        detect beats with low amplitudes (calculated as a fraction of a given peak amplitude percentile)
        """
        if (
            "OMIT_LOW_AMP_PEAKS_PRCTL_ABS" not in params
            or params["OMIT_LOW_AMP_PEAKS_PRCTL_ABS"] is None
        ):
            params["OMIT_LOW_AMP_PEAKS_PRCTL_ABS"] = True
            if params.get("verbose", False):
                print("params.OMIT_LOW_AMP_PEAKS_PRCTL_ABS = true")
            pparams = {}
            if "peak_amps_hist_prctile" not in params:
                pparams["peak_amps_hist_prctile"] = 25.0
                if params.get("verbose", False):
                    print("   pparams.peak_amps_hist_prctile = 25.0")
            else:
                pparams["peak_amps_hist_prctile"] = params["peak_amps_hist_prctile"]

        if params["OMIT_LOW_AMP_PEAKS_PRCTL_ABS"]:
            _, peaks_OMIT_LOW_AMP_PEAKS_PRCTL_ABS = refine_peaks_low_amp_peaks_prctile(
                data_filtered_env_padded,
                peak_indexes_padded,
                "PRCTILE",
                pparams["peak_amps_hist_prctile"],
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_LOW_AMP_PEAKS_PRCTL_ABS = peaks_OMIT_LOW_AMP_PEAKS_PRCTL_ABS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_LOW_AMP_PEAKS_PRCTL_ABS)
            )
            refinement_methods.append("OMIT_LOW_AMP_PEAKS_PRCTL_ABS")

            if (
                "weight_OMIT_LOW_AMP_PEAKS_PRCTL_ABS" not in params
                or params["weight_OMIT_LOW_AMP_PEAKS_PRCTL_ABS"] is None
            ):
                params["weight_OMIT_LOW_AMP_PEAKS_PRCTL_ABS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_LOW_AMP_PEAKS_PRCTL_ABS = {params['weight_OMIT_LOW_AMP_PEAKS_PRCTL_ABS']}"
                    )

            consensus_weights.append(params["weight_OMIT_LOW_AMP_PEAKS_PRCTL_ABS"])

        """
        detect beats that increase average HR variance when omitted (as a bad sign).
        """
        if (
            "OMIT_BEAT_HRV_INCR_BEATS" not in params
            or params["OMIT_BEAT_HRV_INCR_BEATS"] is None
        ):
            params["OMIT_BEAT_HRV_INCR_BEATS"] = True
            if params.get("verbose", False):
                print(
                    f"params.OMIT_BEAT_HRV_INCR_BEATS = {params['OMIT_BEAT_HRV_INCR_BEATS']}"
                )

        if params["OMIT_BEAT_HRV_INCR_BEATS"]:
            mmode = "HEARTRATE"  # 'MORPHOLOGY' or 'HEARTRATE' or 'MORPHOLOGY-HEARTRATE'
            if params.get("verbose", False):
                print(f"   mmode = {mmode}")

            _, peaks_OMIT_BEAT_HRV_INCR_BEATS = refine_peaks_low_snr_beats(
                data_filtered_env_padded,
                peak_indexes_padded,
                mmode,
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_BEAT_HRV_INCR_BEATS = peaks_OMIT_BEAT_HRV_INCR_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_BEAT_HRV_INCR_BEATS)
            )
            refinement_methods.append("OMIT_BEAT_HRV_INCR_BEATS")

            if (
                "weight_OMIT_BEAT_HRV_INCR_BEATS" not in params
                or params["weight_OMIT_BEAT_HRV_INCR_BEATS"] is None
            ):
                params["weight_OMIT_BEAT_HRV_INCR_BEATS"] = 3.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_BEAT_HRV_INCR_BEATS = {params['weight_OMIT_BEAT_HRV_INCR_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_BEAT_HRV_INCR_BEATS"])
        """
        detect beats that increase average beat SNR when omitted (as a good sign of beining an outlier beat)
        """
        if (
            "OMIT_BEAT_SNR_REDUC_BEATS" not in params
            or params["OMIT_BEAT_SNR_REDUC_BEATS"] is None
        ):
            params["OMIT_BEAT_SNR_REDUC_BEATS"] = True
            if params.get("verbose", False):
                print(
                    f"params.OMIT_BEAT_SNR_REDUC_BEATS = {params['OMIT_BEAT_SNR_REDUC_BEATS']}"
                )

        if params["OMIT_BEAT_SNR_REDUC_BEATS"]:
            mmode = (
                "MORPHOLOGY"  # 'MORPHOLOGY' or 'HEARTRATE' or 'MORPHOLOGY-HEARTRATE'
            )
            if params.get("verbose", False):
                print(f"   mmode = {mmode}")

            _, peaks_OMIT_BEAT_SNR_REDUC_BEATS = refine_peaks_low_snr_beats(
                data_filtered_mn_all_channels_padded,
                peak_indexes_padded,
                mmode,
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_BEAT_SNR_REDUC_BEATS = peaks_OMIT_BEAT_SNR_REDUC_BEATS[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack(
                (peaks_all_method, peaks_OMIT_BEAT_SNR_REDUC_BEATS)
            )
            refinement_methods.append("OMIT_BEAT_SNR_REDUC_BEATS")

            if (
                "weight_OMIT_BEAT_SNR_REDUC_BEATS" not in params
                or params["weight_OMIT_BEAT_SNR_REDUC_BEATS"] is None
            ):
                params["weight_OMIT_BEAT_SNR_REDUC_BEATS"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_BEAT_SNR_REDUC_BEATS = {params['weight_OMIT_BEAT_SNR_REDUC_BEATS']}"
                    )

            consensus_weights.append(params["weight_OMIT_BEAT_SNR_REDUC_BEATS"])

        """
        detect beats based on amplitude thresholding (removes below a fraction of the defined percentile)
        """
        if "OMIT_HIGH_STD_AMP" not in params or params["OMIT_HIGH_STD_AMP"] is None:
            params["OMIT_HIGH_STD_AMP"] = True
            if params.get("verbose", False):
                print(f"params.OMIT_HIGH_STD_AMP = {params['OMIT_HIGH_STD_AMP']}")
            pparams = {}
            pparams["k_sigma"] = 4.0
            if params.get("verbose", False):
                print(f"   pparams.k_sigma = {params['k_sigma']}")

        if params["OMIT_HIGH_STD_AMP"]:
            _, peaks_OMIT_HIGH_STD_AMP = refine_peaks_high_amp_std(
                data_filtered_env_padded,
                peak_indexes_padded,
                pparams["k_sigma"],
                params.get("PLOT_DIAGNOSTIC", False),
            )
            peaks_OMIT_HIGH_STD_AMP = peaks_OMIT_HIGH_STD_AMP[
                left_pad_len : left_pad_len + sig_len
            ]

            peaks_all_method = np.vstack((peaks_all_method, peaks_OMIT_HIGH_STD_AMP))
            refinement_methods.append("OMIT_HIGH_STD_AMP")

            if (
                "weight_OMIT_HIGH_STD_AMP" not in params
                or params["weight_OMIT_HIGH_STD_AMP"] is None
            ):
                params["weight_OMIT_HIGH_STD_AMP"] = 1.0
                if params.get("verbose", False):
                    print(
                        f"params.weight_OMIT_HIGH_STD_AMP = {params['weight_OMIT_HIGH_STD_AMP']}"
                    )

            consensus_weights.append(params["weight_OMIT_HIGH_STD_AMP"])

        """
        merge the peaks through consensus
        """
        num_peak_refinement_algorithms = peaks_all_method.shape[0]
        if (
            "NUM_VOTES_TO_KEEP_PEAK" in params
            and "CONSENSUS_WEIGHTS_THRESHOLD" in params
        ):
            if (
                params.get("NUM_VOTES_TO_KEEP_PEAK") is not None
                and params.get("CONSENSUS_WEIGHTS_THRESHOLD") is not None
            ):
                print(
                    "Warning: NUM_VOTES_TO_KEEP_PEAK and CONSENSUS_WEIGHTS_THRESHOLD may not be set together; using only CONSENSUS_WEIGHTS_THRESHOLD"
                )

        if (
            "NUM_VOTES_TO_KEEP_PEAK" not in params
            or params["NUM_VOTES_TO_KEEP_PEAK"] is None
        ):
            if (
                "CONSENSUS_WEIGHTS_THRESHOLD" not in params
                or params["CONSENSUS_WEIGHTS_THRESHOLD"] is None
            ):
                params["CONSENSUS_WEIGHTS_THRESHOLD"] = 0.5  # Default to majority vote
                if params.get("verbose", False):
                    print(
                        f'num_peak_refinement_algorithms = {num_peak_refinement_algorithms}, params.CONSENSUS_WEIGHTS_THRESHOLD = {params["CONSENSUS_WEIGHTS_THRESHOLD"]}'
                    )

        if (
            "CONSENSUS_WEIGHTS_THRESHOLD" in params
            and params["CONSENSUS_WEIGHTS_THRESHOLD"] is not None
        ):
            if not (0 <= params["CONSENSUS_WEIGHTS_THRESHOLD"] <= 1):
                raise ValueError(
                    "params.CONSENSUS_WEIGHTS_THRESHOLD must be between 0 and 1"
                )
            peaks_weighted_average = np.sum(
                np.dot(np.diag(consensus_weights), peaks_all_method), 0
            ) / sum(consensus_weights)
            peak_indexes_consensus = np.where(
                peaks_weighted_average >= params["CONSENSUS_WEIGHTS_THRESHOLD"]
            )[0]
        elif (
            "NUM_VOTES_TO_KEEP_PEAK" in params
            and params["NUM_VOTES_TO_KEEP_PEAK"] is not None
        ):
            if params["NUM_VOTES_TO_KEEP_PEAK"] > num_peak_refinement_algorithms:
                raise ValueError(
                    "Number of required votes to keep a peak exceeds the number of voting algorithms"
                )
            peaks_consensus = np.sum(np.diag(peaks_all_method), axis=0)
            peak_indexes_consensus = np.where(
                peaks_consensus >= params["NUM_VOTES_TO_KEEP_PEAK"]
            )

    else:
        peak_indexes_consensus = peak_indexes
        num_peak_refinement_algorithms = 1

    """
    calculate a likelihood function for the R-peaks (useful for classification and scoring purposes)
    """
    if "likelihood_sigma" not in params or params["likelihood_sigma"] is None:
        params["likelihood_sigma"] = 0.01
        if params.get("verbose", False):
            print(f"params.likelihood_sigma = {params['likelihood_sigma']}")

    if "max_likelihood_span" not in params or params["max_likelihood_span"] is None:
        params["max_likelihood_span"] = 0.1
        if params.get("verbose", False):
            print(f"params.max_likelihood_span = {params['max_likelihood_span']}")

    qrs_likelihoods = np.zeros((num_peak_refinement_algorithms, sig_len))

    for kk in range(num_peak_refinement_algorithms):
        peak_indices = np.where(peaks_all_method[kk, :])[0]  # find non-zero indices
        qrs_likelihoods[kk, :] = peak_surrounding_likelihood(
            sig_len,
            peak_indices,
            fs,
            "GAUSSIAN",
            params["max_likelihood_span"],
            params["likelihood_sigma"],
        )

    qrs_likelihood = np.sum(
        np.diag(consensus_weights) @ qrs_likelihoods, axis=0
    ) / np.sum(consensus_weights)

    """
    replace envelope peaks with original signal peaks, if required
    """
    if params.get("PLOT_DIAGNOSTIC", False):
        tt = np.arange(data.shape[1]) / fs
        plt.figure()

        plt.plot(tt, data.T, label="data")
        plt.plot(tt, data_filtered.T, label="data_filtered")
        plt.plot(tt, data_residual.T, label="data_residual")
        plt.plot(tt, data_filtered_env, label="data_filtered_env")

        plt.plot(
            tt[bumps_indexes],
            data_filtered_env[bumps_indexes],
            "go",
            label="bumps_indexes",
        )
        plt.plot(
            tt[peak_indexes],
            data_filtered_env[peak_indexes],
            "rx",
            markersize=18,
            label="peak_indexes",
        )
        plt.plot(
            tt[peak_indexes_consensus],
            data_filtered_env[peak_indexes_consensus],
            "ko",
            markersize=24,
            label="peak_indexes_consensus",
        )

        plt.legend()
        plt.show()

    if "RETURN_SIGNAL_PEAKS" not in params or params["RETURN_SIGNAL_PEAKS"] is None:
        params["RETURN_SIGNAL_PEAKS"] = True
        if params.get("verbose", False):
            print(f"params.RETURN_SIGNAL_PEAKS = {params['RETURN_SIGNAL_PEAKS']}")

    if params["RETURN_SIGNAL_PEAKS"]:
        if "PEAK_SIGN" not in params:
            params["PEAK_SIGN"] = "AUTO"
            if params.get("verbose", False):
                print(f"params.PEAK_SIGN = {params['PEAK_SIGN']}")
        if "envelope_to_peak_search_wlen" not in params:
            params["envelope_to_peak_search_wlen"] = 0.1
            if params.get("verbose", False):
                print(
                    f"params.envelope_to_peak_search_wlen = {params['envelope_to_peak_search_wlen']}"
                )

        # Calculate window length in samples
        envelope_to_peak_search_wlen = int(
            fs * params["envelope_to_peak_search_wlen"] // 2
        )

        # Process for initial peak indexes
        peak_likelihood_boxes = peak_surrounding_likelihood(
            sig_len, peak_indexes, fs, "BOX", params["envelope_to_peak_search_wlen"], []
        )
        peak_indexes, _ = find_closest_peaks(
            data * np.tile(peak_likelihood_boxes, (data_filtered.shape[0], 1)),
            peak_indexes,
            envelope_to_peak_search_wlen,
            params["PEAK_SIGN"],
            params["PLOT_DIAGNOSTIC"],
        )

        # Process for consensus peak indexes
        peak_likelihood_boxes = peak_surrounding_likelihood(
            sig_len,
            peak_indexes_consensus,
            fs,
            "BOX",
            params["envelope_to_peak_search_wlen"],
            [],
        )
        peak_indexes_consensus, _ = find_closest_peaks(
            data * np.tile(peak_likelihood_boxes, (data_filtered.shape[0], 1)),
            peak_indexes_consensus,
            envelope_to_peak_search_wlen,
            params["PEAK_SIGN"],
            params["PLOT_DIAGNOSTIC"],
        )
    """
    post-extraction peak refinement based on likelihoods
    """
    if (
        "POST_EXT_LIKELIHOOD_BASED_IMPROVEMENT" not in params
        or params["POST_EXT_LIKELIHOOD_BASED_IMPROVEMENT"] is None
    ):
        params["POST_EXT_LIKELIHOOD_BASED_IMPROVEMENT"] = False
        if params.get("verbose", False):
            print(
                f"params.POST_EXT_LIKELIHOOD_BASED_IMPROVEMENT = {params['POST_EXT_LIKELIHOOD_BASED_IMPROVEMENT']}"
            )
    if params["POST_EXT_LIKELIHOOD_BASED_IMPROVEMENT"]:
        likelihood_threshold = 0.4
        if params["verbose"]:
            print(f"   likelihood_threshold = {likelihood_threshold}")

        peak_indexes_consensus = refine_peaks_low_likelihood(
            data_filtered,
            peak_indexes_consensus,
            qrs_likelihood,
            likelihood_threshold,
            rpeak_search_half_wlen,
            params.get("PLOT_DIAGNOSTIC", False),
        )

    if params.get("PLOT_RESULTS", False):
        tt = np.arange(len(data)) / fs

        plt.figure(figsize=(15, 8))

        colors = np.array([0.5, 0.5, 0.5]) + qrs_likelihood * np.array([0, 0, 0.5])
        plt.scatter(
            tt,
            data,
            c=colors,
            s=100,
            edgecolors="none",
            label="QRS likelihood (color-coded from gray to red)",
        )

        plt.plot(tt, data, label="Signal")
        plt.plot(tt, data_filtered, label="Filtered signal")
        plt.plot(tt, data_filtered_env, label="Filtered signal power envelope")

        plt.plot(
            tt[bumps_indexes],
            data_filtered_env[bumps_indexes],
            "g.",
            markersize=14,
            label="Bumps indexes",
        )

        all_marks = ["o", "+", "*", ".", "x", "s", "d", ">", "v", "<", "^", "p", "h"]
        for ll, method in enumerate(refinement_methods):
            pk_indx = np.where(peaks_all_method[ll, :])[0]
            if len(pk_indx) > 0:
                plt.plot(
                    tt[pk_indx],
                    data_filtered_env[pk_indx],
                    all_marks[ll % len(all_marks)],
                    markersize=ll + 10,
                    label=method,
                )

        plt.plot(
            tt[peak_indexes],
            data[peak_indexes],
            "co",
            markersize=16,
            label="Detected R-peaks",
        )
        plt.plot(
            tt[peak_indexes_consensus],
            data[peak_indexes_consensus],
            "ko",
            markerfacecolor="r",
            markersize=20,
            label="Corrected R-peaks",
        )

        plt.grid(True)
        plt.legend(loc="east outside", fontsize=12)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Signal Analysis Overview")
        plt.xlim([tt[0], tt[-1]])
        plt.gca().set_facecolor("white")
        plt.show()

    # make sure there are no replicated peak indexes
    peak_indexes = np.unique(peak_indexes)
    peak_indexes_consensus = np.unique(peak_indexes_consensus)

    # return final peaks
    peaks = np.zeros(sig_len, dtype=int)
    peaks[peak_indexes] = 1

    return peaks, peak_indexes, peak_indexes_consensus, qrs_likelihood
