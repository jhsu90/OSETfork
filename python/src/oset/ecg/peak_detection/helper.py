import numpy as np
import matplotlib.pyplot as plt
from oset.generic.event_stacker import event_stacker

# detect local peaks, which have nearby peaks with absolute higher amplitudes
def find_closest_peaks(data, peak_indexes_candidates, peak_search_half_wlen, operation_mode, plot_results):
    sig_len = len(data)
    polarity = np.sign(data[peak_indexes_candidates]) + 1
    polarity_dominant = np.bincount(polarity.astype(int)).argmax() - 1
    peak_indexes = []

    for jj in range(0, len(peak_indexes_candidates)):
        segment_start = max(0, peak_indexes_candidates[jj] - peak_search_half_wlen)
        segment_end = min(sig_len, peak_indexes_candidates[jj] + peak_search_half_wlen)
        segment = data[segment_start:segment_end]
        segment_first_index = segment_start

        if operation_mode == 'AUTO-BEAT-WISE':
            I_max_min = np.argmax(polarity[jj] * segment)
            pk_indx = I_max_min + segment_first_index
        elif operation_mode == 'AUTO':
            I_max_min = np.argmax(polarity_dominant * segment)
            pk_indx = I_max_min + segment_first_index
        elif operation_mode == 'POS':
            I_max = np.argmax(segment)
            pk_indx = I_max + segment_first_index
        elif operation_mode == 'NEG':
            I_min = np.argmin(segment)
            pk_indx = I_min + segment_first_index
        else:
            raise ValueError('Undefined peak sign detection mode.')

        peak_indexes.append(pk_indx)

    peaks = np.zeros(sig_len)
    peaks[peak_indexes] = 1

    if plot_results:
        plt.figure(figsize=(16, 8))
        n = np.arange(1, len(data) + 1)
        plt.plot(n, data, label='data')
        plt.scatter(n[peak_indexes_candidates], data[peak_indexes_candidates], color='g', s=100, marker='x',
                    label='input peaks')
        plt.scatter(n[peak_indexes], data[peak_indexes], color='r', s=100, marker='o', label='refined peaks')
        plt.legend(loc='best')
        plt.title('find_closest_peaks')
        plt.grid(True)
        plt.show()

    return peak_indexes, peaks

# detect lower-amplitude peaks within a minimal window size
def refine_peaks_too_close_low_amp(data, peak_indexes_candidates, peak_search_half_wlen, mode, plot_results):
    sig_len = len(data)
    peak_indexes = []
    for jj in range(0,len(peak_indexes_candidates)):
        pk_index_candidate = peak_indexes_candidates[jj]
        segment_start = max(0, pk_index_candidate - peak_search_half_wlen)
        segment_end = min(sig_len, pk_index_candidate + peak_search_half_wlen)
        segment = data[segment_start:segment_end]

        if mode == 'POS':
            if max(segment) == data[pk_index_candidate]:
                peak_indexes.append(pk_index_candidate)
        elif mode == 'NEG':
            if min(segment) == data[pk_index_candidate]:
                peak_indexes.append(pk_index_candidate)
        else:
            raise ValueError('Undefined peak sign detection mode.')
    peaks = np.zeros(sig_len)
    peaks[peak_indexes] = 1

    if plot_results:
        plt.figure(figsize=(16, 8))
        n = np.arange(1, len(data) + 1)
        plt.plot(n, data, label='data')
        plt.scatter(n[peak_indexes_candidates], data[peak_indexes_candidates], color='g', s=100, marker='x',
                    label='input peaks')
        plt.scatter(n[peak_indexes], data[peak_indexes], color='r', s=100, marker='o', label='refined peaks')
        plt.legend(loc='best')
        plt.title('refine_peaks_too_close_low_amp')
        plt.grid(True)
        plt.show()
    return peak_indexes, peaks

# detect beats based on amplitude thresholding (removes below the given percentile)
def refine_peaks_low_amp_peaks_prctile(data_env, peak_indexes, method, pparam, plot_results):
    if method == 'PRCTILE':
        percentile = pparam
        bumps_amp_threshold = np.percentile(data_env[peak_indexes], percentile)
    elif method == 'LEVEL':
        bumps_amp_threshold = pparam

    peak_indexes_refined = [idx for idx in peak_indexes if data_env[idx] >= bumps_amp_threshold]

    peaks = np.zeros(len(data_env))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data_env) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data_env, label='data')
        plt.scatter(n[peak_indexes], data_env[peak_indexes], color='green', s=100, marker='x', label='input peaks')
        plt.scatter(n[peak_indexes_refined], data_env[peak_indexes_refined], color='red', s=100, marker='o',
                    label='refined peaks')
        plt.legend(loc='best')
        plt.title('refine_peaks_low_amp_peaks_prctile')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks
# detect beats based on amplitude thresholding (removes below a fraction of the defined percentile)
def refine_peaks_low_amp_peaks_prctile_fraction(data, peak_indexes, pparams, plot_results):
    peak_indexes_refined = peak_indexes.copy()
    peak_amps = data[peak_indexes]
    threshold = pparams['percentile_fraction'] * np.percentile(peak_amps, pparams['percentile'])
    I_omit = peak_amps < threshold
    peak_indexes_refined = np.delete(peak_indexes_refined, np.where(I_omit))

    peaks = np.zeros(len(data))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data, label='data')
        plt.scatter(n[peak_indexes], data[peak_indexes], color='green', marker='x', markersize=18, label='input peaks')
        plt.scatter(n[peak_indexes_refined], data[peak_indexes_refined], color='red', marker='o', markersize=18, label='refined peaks')
        plt.legend(loc='best')
        plt.title('refine_peaks_low_amp_peaks_prctile_fraction')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
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
        plt.plot(n, data, label='Data')
        plt.scatter(n[peak_indexes], data[peak_indexes], color='green', marker='x', markersize=18, label='Input Peaks')
        plt.scatter(n[peak_indexes_refined], data[peak_indexes_refined], color='red', marker='o', markersize=18,
                    label='Refined Peaks')
        plt.legend(loc='best')
        plt.title('refine_peaks_high_amp_std')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks

# detect beats based on waveform similarity
def refine_peaks_waveform_similarity(data, peak_indexes, pparams, method, plot_results):
    event_width = 2 * int(np.round(np.median(np.diff(peak_indexes)) / 2)) + 1
    stacked_beats = event_stacker()

    #    (np.array([data[idx - event_width // 2: idx + event_width // 2 + 1] for idx in peak_indexes]))

    if method == 'CORR':
        rho_beats = np.dot(stacked_beats, stacked_beats.T)
        avg_beat_corr_with_others = np.median(np.nan_to_num(rho_beats) + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0)
        threshold = pparams['percentile_fraction'] * np.percentile(avg_beat_corr_with_others, pparams['percentile'])

    elif method == 'CORRCOEF':
        rho_beats = np.corrcoef(stacked_beats)
        avg_beat_corr_with_others = np.median(np.nan_to_num(rho_beats) + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0)
        threshold = pparams['percentile_fraction'] * np.percentile(avg_beat_corr_with_others, pparams['percentile'])

    elif method == 'ABS-CORRCOEF':
        rho_beats = np.corrcoef(stacked_beats)
        avg_beat_corr_with_others = np.mean(np.nan_to_num(rho_beats) + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0)
        threshold = pparams['beat_corrcoef_th']

    elif method == 'NEG-CORR':
        rho_beats = np.corrcoef(stacked_beats)
        avg_beat_corr_with_others = np.mean(np.nan_to_num(rho_beats) + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0)
        threshold = 0

    elif method == 'BEAT-STD':
        rho_beats = np.corrcoef(stacked_beats)
        avg_beat_corr_with_others = np.mean(np.nan_to_num(rho_beats) + np.diag(np.full(rho_beats.shape[0], np.nan)), axis=0)
        threshold = np.mean(avg_beat_corr_with_others) + pparams['k_sigma'] * np.std(avg_beat_corr_with_others)

    else:
        raise ValueError('undefined mode')

    # Apply the threshold to determine which beats to omit
    I_omit = avg_beat_corr_with_others < threshold

    peak_indexes_refined = np.delete(peak_indexes, np.where(I_omit))
    peaks = np.zeros(len(data))
    peaks[peak_indexes_refined] = 1

    if plot_results:
        n = np.arange(1, len(data) + 1)
        plt.figure(figsize=(16, 8))
        plt.plot(n, data, label='data')
        plt.scatter(n[peak_indexes], data[peak_indexes], color='green', marker='x', markersize=18, label='input peaks')
        plt.scatter(n[peak_indexes_refined], data[peak_indexes_refined], color='red', marker='o', markersize=18,
                    label='refined peaks')
        plt.legend(loc='best')
        plt.title(f'refine_peaks_waveform_similarity: {method}')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    return peak_indexes_refined, peaks

# Generate a sample signal
t = np.linspace(0, 3 * np.pi, 200)
signal = np.sin(t) + 0.8 * np.cos(2*t)

peak_indexes_candidates = [50, 100, 150]

# Settings
fs = 1  # Sample frequency, not relevant here as no real-time unit is considered
peak_search_half_wlen = 2  # Look 10 samples around each candidate for the true peak
operation_mode = 'POS'  # We know we're looking for positive peaks in a sine wave
plot_results = True  # Enable plotting to visually inspect the results

# Function call
peak_indexes, peaks = refine_peaks_too_close_low_amp(signal, peak_indexes_candidates, peak_search_half_wlen, operation_mode, plot_results)

# Print the outputs for review
print("Refined Peak Indexes:", peak_indexes)
