import numpy as np
import matplotlib.pyplot as plt
from oset.generic.event_stacker import event_stacker
from oset.generic.lp_filter.lp_filter_zero_phase import lp_filter_zero_phase
from oset.ecg.peak_detection.robust_weighted_average import robust_weighted_average
from scipy.signal import convolve


# detect local peaks, which have nearby peaks with absolute higher amplitudes
def find_closest_peaks(
    data, peak_indexes_candidates, peak_search_half_wlen, operation_mode, plot_results
):
    sig_len = len(data)
    polarity = np.sign(data[peak_indexes_candidates]) + 1
    polarity_dominant = np.bincount(polarity.astype(int)).argmax() - 1
    peak_indexes = []

    for jj in range(0, len(peak_indexes_candidates)):
        segment_start = max(0, peak_indexes_candidates[jj] - peak_search_half_wlen)
        segment_end = min(sig_len, peak_indexes_candidates[jj] + peak_search_half_wlen)
        segment = data[segment_start:segment_end]
        segment_first_index = segment_start

        if operation_mode == "AUTO-BEAT-WISE":
            I_max_min = np.argmax(polarity[jj] * segment)
            pk_indx = I_max_min + segment_first_index
        elif operation_mode == "AUTO":
            I_max_min = np.argmax(polarity_dominant * segment)
            pk_indx = I_max_min + segment_first_index
        elif operation_mode == "POS":
            I_max = np.argmax(segment)
            pk_indx = I_max + segment_first_index
        elif operation_mode == "NEG":
            I_min = np.argmin(segment)
            pk_indx = I_min + segment_first_index
        else:
            raise ValueError("Undefined peak sign detection mode.")

        peak_indexes.append(pk_indx)

    peaks = np.zeros(sig_len)
    peaks[peak_indexes] = 1

    if plot_results:
        plt.figure(figsize=(16, 8))
        n = np.arange(1, len(data) + 1)
        plt.plot(n, data, label="data")
        plt.scatter(
            n[peak_indexes_candidates],
            data[peak_indexes_candidates],
            color="g",
            s=100,
            marker="x",
            label="input peaks",
        )
        plt.scatter(
            n[peak_indexes],
            data[peak_indexes],
            color="r",
            s=100,
            marker="o",
            label="refined peaks",
        )
        plt.legend(loc="best")
        plt.title("find_closest_peaks")
        plt.grid(True)
        plt.show()

    return peak_indexes, peaks


# detect lower-amplitude peaks within a minimal window size
def refine_peaks_too_close_low_amp(
    data, peak_indexes_candidates, peak_search_half_wlen, mode, plot_results
):
    sig_len = len(data)
    peak_indexes = []
    for jj in range(0, len(peak_indexes_candidates)):
        pk_index_candidate = peak_indexes_candidates[jj]
        segment_start = max(0, pk_index_candidate - peak_search_half_wlen)
        segment_end = min(sig_len, pk_index_candidate + peak_search_half_wlen)
        segment = data[segment_start:segment_end]

        if mode == "POS":
            if max(segment) == data[pk_index_candidate]:
                peak_indexes.append(pk_index_candidate)
        elif mode == "NEG":
            if min(segment) == data[pk_index_candidate]:
                peak_indexes.append(pk_index_candidate)
        else:
            raise ValueError("Undefined peak sign detection mode.")
    peaks = np.zeros(sig_len)
    peaks[peak_indexes] = 1

    if plot_results:
        plt.figure(figsize=(16, 8))
        n = np.arange(1, len(data) + 1)
        plt.plot(n, data, label="data")
        plt.scatter(
            n[peak_indexes_candidates],
            data[peak_indexes_candidates],
            color="g",
            s=100,
            marker="x",
            label="input peaks",
        )
        plt.scatter(
            n[peak_indexes],
            data[peak_indexes],
            color="r",
            s=100,
            marker="o",
            label="refined peaks",
        )
        plt.legend(loc="best")
        plt.title("refine_peaks_too_close_low_amp")
        plt.grid(True)
        plt.show()
    return peak_indexes, peaks


# detect beats based on amplitude thresholding (removes below the given percentile)
def refine_peaks_low_amp_peaks_prctile(
    data_env, peak_indexes, method, pparam, plot_results
):
    if method == "PRCTILE":
        percentile = pparam
        bumps_amp_threshold = np.percentile(data_env[peak_indexes], percentile)
    elif method == "LEVEL":
        bumps_amp_threshold = pparam
    else:
        raise ValueError("undefined method")

    peak_indexes_refined = [
        idx for idx in peak_indexes if data_env[idx] >= bumps_amp_threshold
    ]

    peaks = np.zeros(len(data_env))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data_env) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data_env, label="data")
        plt.scatter(
            n[peak_indexes],
            data_env[peak_indexes],
            color="green",
            s=100,
            marker="x",
            label="input peaks",
        )
        plt.scatter(
            n[peak_indexes_refined],
            data_env[peak_indexes_refined],
            color="red",
            s=100,
            marker="o",
            label="refined peaks",
        )
        plt.legend(loc="best")
        plt.title("refine_peaks_low_amp_peaks_prctile")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# detect beats based on amplitude thresholding (removes below a fraction of the defined percentile)
def refine_peaks_low_amp_peaks_prctile_fraction(
    data, peak_indexes, pparams, plot_results
):
    peak_indexes_refined = peak_indexes.copy()
    peak_amps = data[peak_indexes]
    threshold = pparams["percentile_fraction"] * np.percentile(
        peak_amps, pparams["percentile"]
    )
    I_omit = peak_amps < threshold
    peak_indexes_refined = np.delete(peak_indexes_refined, np.where(I_omit))

    peaks = np.zeros(len(data))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data, label="data")
        plt.scatter(
            n[peak_indexes],
            data[peak_indexes],
            color="green",
            marker="x",
            markersize=18,
            label="input peaks",
        )
        plt.scatter(
            n[peak_indexes_refined],
            data[peak_indexes_refined],
            color="red",
            marker="o",
            markersize=18,
            label="refined peaks",
        )
        plt.legend(loc="best")
        plt.title("refine_peaks_low_amp_peaks_prctile_fraction")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# detect beats based on amplitude thresholding (removes below a fraction of the defined percentile)
def refine_peaks_high_amp_std(data, peak_indexes, k_sigma, plot_results):
    peak_indexes_refined = np.array(peak_indexes)
    peak_amps = data[peak_indexes]
    mean_amp = np.mean(peak_amps)
    std_amp = np.std(peak_amps)

    I_omit = np.abs(peak_amps - mean_amp) > k_sigma * std_amp
    peak_indexes_refined = np.delete(peak_indexes_refined, np.where(I_omit))

    peaks = np.zeros(len(data))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data, label="Data")
        plt.scatter(
            n[peak_indexes],
            data[peak_indexes],
            color="green",
            marker="x",
            markersize=18,
            label="Input Peaks",
        )
        plt.scatter(
            n[peak_indexes_refined],
            data[peak_indexes_refined],
            color="red",
            marker="o",
            markersize=18,
            label="Refined Peaks",
        )
        plt.legend(loc="best")
        plt.title("refine_peaks_high_amp_std")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# detect beats based on waveform similarity
def refine_peaks_waveform_similarity(data, peak_indexes, pparams, method, plot_results):
    event_width = 2 * int(np.round(np.median(np.diff(peak_indexes)) / 2)) + 1
    stacked_beats, _ = event_stacker(data, peak_indexes, event_width)

    # stacked_beats = (np.array([data[idx - event_width // 2: idx + event_width // 2 + 1] for idx in peak_indexes]))

    if method == "CORR":
        rho_beats = np.dot(stacked_beats, stacked_beats.T)
        avg_beat_corr_with_others = np.nanmedian(
            rho_beats + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0
        )
        threshold = pparams["percentile_fraction"] * np.percentile(
            avg_beat_corr_with_others, pparams["percentile"]
        )

    elif method == "CORRCOEF":
        rho_beats = np.corrcoef(stacked_beats.T)
        avg_beat_corr_with_others = np.nanmedian(
            rho_beats + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0
        )
        threshold = pparams["percentile_fraction"] * np.percentile(
            avg_beat_corr_with_others, pparams["percentile"]
        )

    elif method == "ABS-CORRCOEF":
        rho_beats = np.corrcoef(stacked_beats.T)
        avg_beat_corr_with_others = np.nanmean(
            rho_beats + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0
        )
        threshold = pparams["beat_corrcoef_th"]

    elif method == "NEG-CORR":
        rho_beats = np.corrcoef(stacked_beats.T)
        avg_beat_corr_with_others = np.nanmean(
            rho_beats + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0
        )
        threshold = 0

    elif method == "BEAT-STD":
        rho_beats = np.corrcoef(stacked_beats.T)
        avg_beat_corr_with_others = np.nanmean(
            rho_beats + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0
        )
        threshold = np.mean(avg_beat_corr_with_others) + pparams["k_sigma"] * np.std(
            avg_beat_corr_with_others
        )

    else:
        raise ValueError("undefined mode")

    # Apply the threshold to determine which beats to omit
    I_omit = avg_beat_corr_with_others < threshold

    peak_indexes_refined = np.delete(peak_indexes, np.where(I_omit))
    peaks = np.zeros(len(data))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data, label="data")
        plt.scatter(
            n[peak_indexes],
            data[peak_indexes],
            color="green",
            marker="x",
            markersize=18,
            label="input peaks",
        )
        plt.scatter(
            n[peak_indexes_refined],
            data[peak_indexes_refined],
            color="red",
            marker="o",
            markersize=18,
            label="refined peaks",
        )
        plt.legend(loc="best")
        plt.title(f"refine_peaks_waveform_similarity: {method}")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# detect low-power beats
def refine_peaks_low_power_beats(
    data, peak_indexes, max_amp_prctile, beat_std_med_frac_th, plot_results
):
    event_width = 2 * round(np.median(np.diff(peak_indexes)) / 2) + 1
    stacked_beats, num_non_zeros = event_stacker(data, peak_indexes, event_width)
    std_beats = (
        (event_width - 1) * np.std(stacked_beats, axis=1, ddof=1) / (num_non_zeros - 1)
    )

    I_omit = std_beats < beat_std_med_frac_th * np.percentile(
        std_beats, max_amp_prctile
    )
    peak_indexes_refined = np.delete(peak_indexes, np.where(I_omit))

    peaks = np.zeros(len(data), dtype=int)
    peaks[peak_indexes_refined] = 1

    if plot_results:
        plt.figure(figsize=(16, 6))
        plt.plot(data, label="data")
        plt.plot(
            peak_indexes, data[peak_indexes], "gx", markersize=9, label="input peaks"
        )
        plt.plot(
            peak_indexes_refined,
            data[peak_indexes_refined],
            "ro",
            markersize=9,
            label="refined peaks",
        )
        plt.legend(loc="best")
        plt.title("Refine Peaks Low Power Beats")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# peak refinement based on likelihoods
def refine_peaks_low_likelihood(
    data,
    peak_indexes,
    qrs_likelihood,
    likelihood_threshold,
    peak_search_half_wlen,
    plot_results,
):
    sig_len = len(data)
    signal_abs = np.sqrt(np.mean(data**2, axis=0))

    if np.max(signal_abs) <= 0:
        raise ValueError("Empty data")

    signal_likelihood = qrs_likelihood * (signal_abs / np.max(signal_abs))
    bumps_indexes = np.where(signal_likelihood >= likelihood_threshold)[0]
    peak_indexes_refined = []

    for index in bumps_indexes:
        segment_start = max(1, index - peak_search_half_wlen)
        segment_end = min(sig_len, index + peak_search_half_wlen)
        segment = range(segment_start, segment_end + 1)  # +1 to make it inclusive
        if np.max(signal_likelihood[segment]) == signal_likelihood[index]:
            peak_indexes_refined.append(index)

    peak_indexes_refined = np.intersect1d(peak_indexes_refined, peak_indexes)

    peaks = np.zeros(sig_len)
    peaks[peak_indexes_refined] = 1

    if plot_results:
        plt.figure(figsize=(16, 6))
        plt.plot(data, label="Data")
        plt.plot(
            peak_indexes, data[peak_indexes], "gx", markersize=9, label="Input Peaks"
        )
        plt.plot(
            peak_indexes_refined,
            data[peak_indexes_refined],
            "ro",
            markersize=9,
            label="Refined Peaks",
        )
        plt.legend(loc="best")
        plt.title("Refine Peaks Low Likelihood")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# detect beats that decrease average beat SNR and increase HR variance.
def refine_peaks_low_snr_beats(data, peak_indexes, mmode, plot_results):
    max_itr = len(peak_indexes)
    event_width = 2 * round(np.median(np.diff(peak_indexes)) / 2) + 1
    stacked_beats, num_non_zeros = event_stacker(data, peak_indexes, event_width)
    ECG_robust_mean = robust_weighted_average(
        stacked_beats
    )  # robust_weighted_average under construction
    # ECG_robust_mean = np.mean(stacked_beats, axis=1)
    ECG_robust_mean_replicated = np.ones(len(peak_indexes), dtype=int) * ECG_robust_mean
    noise = stacked_beats - ECG_robust_mean_replicated
    snr_initial = (
        20
        * np.log10(
            np.linalg.norm(
                ECG_robust_mean_replicated,
            )
        )
        / np.linalg.norm(noise)
    )

    included_indexes = list(range((peak_indexes)))
    for itr in range(max_itr):
        num_includes_beats = len(peak_indexes[included_indexes])
        rr_std = np.std(np.diff(peak_indexes[included_indexes]))
        snr_excluding_this_beat = np.zeros((1, num_includes_beats))
        rr_std_excluding_this_beat = np.zeros((1, num_includes_beats))
        for p in range(num_includes_beats):
            all_included_indexes_but_this_beat = included_indexes.copy()
            all_included_indexes_but_this_beat.remove(
                all_included_indexes_but_this_beat[p]
            )

            if all_included_indexes_but_this_beat:
                ECG_robust_mean = robust_weighted_average(
                    stacked_beats[all_included_indexes_but_this_beat, :]
                )  # robust_weighted_average under construction
                # ECG_robust_mean = np.mean(
                #     stacked_beats[all_included_indexes_but_this_beat, :], axis=0
                # )
            else:
                ECG_robust_mean = np.mean(
                    stacked_beats[all_included_indexes_but_this_beat, :], axis=0
                )

            ECG_robust_mean_replicated = np.tile(
                ECG_robust_mean, (len(all_included_indexes_but_this_beat), 1)
            )
            noise = (
                stacked_beats[all_included_indexes_but_this_beat, :]
                - ECG_robust_mean_replicated
            )
            snr_excluding_this_beat[p] = 20 * np.log10(
                np.linalg.norm(ECG_robust_mean_replicated) / np.linalg.norm(noise)
            )
            rr_std_excluding_this_beat[p] = np.std(
                np.diff(peak_indexes[all_included_indexes_but_this_beat])
            )

        snr_excluding_worse_beat = np.max(snr_excluding_this_beat)
        I_worse_beat = np.argmax(snr_excluding_this_beat)
        if mmode == "MORPHOLOGY":
            if snr_excluding_worse_beat > snr_initial:
                included_indexes.remove(included_indexes[I_worse_beat])
            else:
                break

        elif mmode == "HEARTRATE":
            if rr_std_excluding_this_beat[I_worse_beat] < rr_std:
                included_indexes.remove(included_indexes[I_worse_beat])
            else:
                break

        elif mmode == "MORPHOLOGY-HEARTRATE":
            if (
                snr_excluding_worse_beat > snr_initial
                and rr_std_excluding_this_beat[I_worse_beat] < rr_std
            ):
                included_indexes.remove(included_indexes[I_worse_beat])
            else:
                break

    peak_indexes_refined = [peak_indexes[i] for i in included_indexes]
    peaks = np.zeros(len(data), dtype=int)
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(len(data))
        plt.figure(figsize=(12, 6))
        plt.plot(n, data, label="Data")
        plt.scatter(
            n[peak_indexes],
            data[peak_indexes],
            color="g",
            s=100,
            marker="x",
            label="Input Peaks",
        )
        plt.scatter(
            n[peak_indexes_refined],
            data[peak_indexes_refined],
            color="r",
            s=100,
            marker="o",
            label="Refined Peaks",
        )
        plt.legend(loc="best")
        plt.title("Refine Peaks Low SNR Beats")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Amplitude")
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks


# detect beats that were most impacted by preprocessing filter (for T-wave detection purposes)
def refine_peaks_filter_energy_impacted_beats(
    data_pre_filter, data_post_filter, peak_indexes, pparams
):
    # Apply lowpass and highpass filtering if required
    if "ff_low_cutoff" in pparams and pparams["ff_low_cutoff"] is not None:
        data_pre_filter -= lp_filter_zero_phase(
            data_pre_filter, pparams["ff_low_cutoff"]
        )
        data_post_filter -= lp_filter_zero_phase(
            data_post_filter, pparams["ff_low_cutoff"]
        )

    if "ff_high_cutoff" in pparams and pparams["ff_high_cutoff"] is not None:
        data_pre_filter = lp_filter_zero_phase(
            data_pre_filter, pparams["ff_high_cutoff"]
        )
        data_post_filter = lp_filter_zero_phase(
            data_post_filter, pparams["ff_high_cutoff"]
        )

    # Stack the pre- and post-filter beats
    event_width = 2 * int(np.median(np.diff(peak_indexes)) / 2) + 1
    stacked_beats_pre_filter, _ = event_stacker(
        data_pre_filter, peak_indexes, event_width
    )
    stacked_beats_post_filter, _ = event_stacker(
        data_post_filter, peak_indexes, event_width
    )

    # Calculate the power ratios of the beats before vs after filtering
    pre_filter_beat_vars = np.var(stacked_beats_pre_filter, axis=1, ddof=1)
    post_filter_beat_vars = np.var(stacked_beats_post_filter, axis=1, ddof=1)
    pre_to_post_power_ratio = pre_filter_beat_vars / post_filter_beat_vars

    I_include = (
        pre_to_post_power_ratio > pparams["pre_to_post_filter_power_ratio_th"]
    ) & (
        post_filter_beat_vars
        > pparams["percentile_fraction"]
        * np.percentile(post_filter_beat_vars, pparams["percentile"])
    )

    peak_indexes_refined = peak_indexes[I_include]
    peaks = np.zeros(len(data_pre_filter))
    peaks[peak_indexes_refined] = 1

    return peak_indexes_refined, peaks


# matched filter using average beat shape
def signal_specific_matched_filter(data, peak_indexes):
    if len(peak_indexes) > 1:
        event_width = 2 * round(np.median(np.diff(peak_indexes)) / 2) + 1
        sig_len = data.shape[1]
        data_enhanced = np.zeros_like(data)

        for ch in range(data.shape[0]):
            stacked_beats, _ = event_stacker(data[ch, :], peak_indexes, event_width)
            robust_mean = robust_weighted_average(
                stacked_beats
            )  # robust_weighted_average() under construction
            matched_filter_out = convolve(robust_mean[::-1], data[ch, :], mode="full")
            lag = round(len(robust_mean) / 2)
            data_enhanced[ch, :] = matched_filter_out[lag : sig_len + lag]
            data_enhanced[ch, :] = (
                np.std(data[ch, :]) / np.std(data_enhanced[ch, :])
            ) * data_enhanced[ch, :]
    else:
        data_enhanced = data.copy()

    data_enhanced_env = np.sqrt(np.sum(data_enhanced**2, axis=0))
    return data_enhanced, data_enhanced_env


# likelihood of peaks as we move towards or away a peak
def peak_surrounding_likelihood(
    sig_len, peak_indexes, fs, method, max_span, likelihood_sigma
):
    peaks = np.zeros(sig_len)
    peaks[peak_indexes] = 1
    half_span = round(fs * max_span / 2)

    if method == "GAUSSIAN":
        tt = np.arange(-half_span, half_span + 1) / fs
        template = np.exp(-(tt**2) / (2 * likelihood_sigma**2))
    elif method == "BOX":
        template = np.ones(2 * half_span + 1)
    else:
        raise ValueError("Undefined method")

    lag = half_span
    qrs_likelihood = convolve(peaks, template, mode="full")
    qrs_likelihood = qrs_likelihood[lag : lag + sig_len]

    if np.max(qrs_likelihood) > 1:
        qrs_likelihood /= np.max(qrs_likelihood)

    return qrs_likelihood
