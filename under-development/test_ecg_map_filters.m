% Test script for an ECG denoiser based on a data driven MAP estimator in
% the phase and time domains

close all
clear
clc

seg_len = 10.0; % ECG segment length

% Load data
% datafilepath = '../../../DataFiles/Physionet.org/files/ptbdb/1.0.0/'; % Change this path to where you have the .mat data files
% fs = 1000.0; % Sampling frequency of the data (put it in the loop and read it from the data if not fixed across all records)
% % % % ofname = 'PTBAnalysisResults.csv';

datafilepath = '../../../DataFiles/Physionet.org/files/qtdb/1.0.0/'; % Change this path to where you have the .mat data files
fs = 250.0; % Sampling frequency of the data (put it in the loop and read it from the data if not fixed across all records)

ofname = './GPFilteringResults.csv';

filelist = dir(fullfile([datafilepath, '**/*.mat']));  % get list of all mat files of interest
% Make N by 2 matrix of fieldname + value type
variable_names_types = [["FileID", "string"]; ...
    ["Channel", "int32"]; ...
    ["First_index", "int32"]; ...
    ["Last_index", "int32"]; ...
    ["NumRPeaks", "int32"]; ...
    ["Run", "int32"]; ...
    ["SetInSNR", "double"]; ...
    ["ActInSNR", "double"]; ...
    ["SignalVar", "double"]; ...
    ["NoiseVar", "double"]; ...
    ["OutSNR_PriorPhaseBased", "double"]; ...
    ["OutSNR_PostPhaseBased", "double"]; ...
    ["OutSNR_PriorTimeBased", "double"]; ...
    ["OutSNR_PostTimeBased", "double"]];

% Make table using fieldnames & value types from above
ResultsTableEmpty = table('Size',[1,size(variable_names_types,1)],...
    'VariableNames', variable_names_types(:,1),...
    'VariableTypes', variable_names_types(:,2));

ResultsTable = ResultsTableEmpty;
% Baseline wander removal filter
w1 = 0.72; % First stage baseline wander removal window size (in seconds)
w2 = 0.87; % Second stage baseline wander removal window size (in seconds)
BASELINE_REMOVAL_APPROACH = 'BP'; %'MDMN'; % baseline wander removal method
seg_len_samples = (fs*seg_len);
OneRowSet = false;
for k = 1 : length(filelist) % Sweep over all or specific records
    datafilename = [filelist(k).folder '/' filelist(k).name];
    data0 = load(datafilename);
    data0 = data0.val;


    num_segments = floor(size(data0, 2) / seg_len_samples);
    for segment = 1 : num_segments
        first_sample = (segment-1)*seg_len_samples + 1;
        last_sample = segment*seg_len_samples;

        data = data0(:, first_sample:last_sample); % select a short segment

        switch(BASELINE_REMOVAL_APPROACH)
            case 'BP'
                data = data - lp_filter_zero_phase(data, 5.0/fs);
                data = lp_filter_zero_phase(data, 80.0/fs);
            case 'MDMN'
                wlen1 = round(w1 * fs);
                wlen2 = round(w2 * fs);
                for jj = 1 : size(data, 1)
                    bl1 = BaseLine1(data(jj, :), wlen1, 'md');
                    data(jj, :) = data(jj, :) - BaseLine1(bl1, wlen2, 'mn');
                end
            otherwise
                warning('Unknown method. Bypassing baseline wander removal.');
        end

        for SNR_pre_set = 30 : 5 : 30 % the desired input SNR
            % for SNR_pre_set = 55 : 5 : 60 % the desired input SNR
            for ch = 1 : size(data, 1) % sweep over all or a single desired channel
                for itr = 1 : 1 % number of repeated runs
                    sig = data(ch, :);
                    sd = sqrt(var(sig) / 10^(SNR_pre_set/10));
                    noise = sd * randn(size(sig));
                    x = sig + noise;

                    if 0
                        % f0 = 1.0; % approximate heart rate (in Hz) used for R-peak detection
                        % [peaks, peaks_indexes] = peak_det_local_search(sig, f0/fs);                  % peak detection
                    else
                        % peak_detector_params.saturate = 1;
                        % peak_detector_params.hist_search_th = 0.9;
                        % peak_detector_params.rpeak_search_wlen = 0.3; % MAX detectable HR (in BPM) = 60/rpeak_search_wlen
                        % peak_detector_params.filter_type = 'MULT_MATCHED_FILTER';%'BANDPASS_FILTER', 'MATCHED_FILTER', 'MULT_MATCHED_FILTER'
                        % [peaks, peaks_indexes] = PeakDetectionProbabilistic(sig, fs, peak_detector_params);
                        [peaks, peaks_indexes] = peak_det_likelihood(sig, fs);
                    end
                    % GPfilterparams.bins = 350               ;              %300; % number of phase domain bins
                    GPfilterparams.BEAT_AVG_METHOD = 'MEDIAN'; % 'MEAN' or 'MEDIAN'
                    GPfilterparams.NOISE_VAR_EST_METHOD = 'AVGLOWER'; %'MIN', 'AVGLOWER', 'MEDLOWER', 'PERCENTILE'
                    GPfilterparams.p = 0.5;
                    GPfilterparams.avg_bins = 10;
                    GPfilterparams.SMOOTH_PHASE = 'GAUSSIAN';%'BYPASS';%'GAUSSIAN';
                    GPfilterparams.gaussianstd = 1.0;
                    %                 GPfilterparams.SMOOTH_PHASE = 'MA';
                    %                 GPfilterparams.wlen_phase = 3;
                    %                 GPfilterparams.wlen_time = 3;
                    GPfilterparams.plotresults = 0;
                    GPfilterparams.nvar_factor = 1.0; % noise variance over/under estimation factor (1 by default)
                    GPfilterparams.interp_order = 2;

                    GPfilterparams.nvar = 1*var(noise);
                    
                    GPfilterparams.SMOOTH_COV = 'GAUSSIAN'; %'BYPASS';%'GAUSSIAN';
                    GPfilterparams.smooth_cov_gaussianstd = 3.0;

                    [data_posterior_est_phase_based_fullcov, data_prior_est_phase_based_fullcov] = ecg_den_phase_domain_gp_full_cov(x, peaks_indexes, GPfilterparams);
                    [data_posterior_est_phase_based, data_prior_est_phase_based] = ecg_den_phase_domain_gp(x, peaks, GPfilterparams);
                    [data_posterior_est_time_based, data_prior_est_time_based] = ecg_den_time_domain_gp(x, peaks, GPfilterparams);

                    seg_ind = round(length(sig)/3) : round(2*length(sig)/3);
                    s_power = mean(sig(seg_ind).^2);
                    noise_power = mean(noise(seg_ind).^2);
                    SNR_pre = 10 * log10(s_power / noise_power);
                    SNR_prior_phase_based = 10 * log10(s_power / mean((data_prior_est_phase_based(seg_ind) - sig(seg_ind)).^2));
                    SNR_posterior_phase_based = 10 * log10(s_power / mean((data_posterior_est_phase_based(seg_ind) - sig(seg_ind)).^2));
                    SNR_prior_phase_based_fullcov = 10 * log10(s_power / mean((data_prior_est_phase_based_fullcov(seg_ind) - sig(seg_ind)).^2));
                    SNR_posterior_phase_based_fullcov = 10 * log10(s_power / mean((data_posterior_est_phase_based_fullcov(seg_ind) - sig(seg_ind)).^2));
                    SNR_prior_time_based = 10 * log10(s_power / mean((data_prior_est_time_based(seg_ind) - sig(seg_ind)).^2));
                    SNR_posterior_time_based = 10 * log10(s_power / mean((data_posterior_est_time_based(seg_ind) - sig(seg_ind)).^2));

                    % Log results
                    TableRow = ResultsTableEmpty;
                    TableRow.FileID = string(filelist(k).name);
                    TableRow.Channel = ch;
                    TableRow.First_index = first_sample;
                    TableRow.Last_index = last_sample;
                    TableRow.NumRPeaks = length(peaks_indexes);
                    TableRow.Run = itr;
                    TableRow.SetInSNR = SNR_pre_set;
                    TableRow.ActInSNR = SNR_pre;
                    TableRow.SignalVar = s_power;
                    TableRow.NoiseVar = noise_power;
                    TableRow.OutSNR_PriorPhaseBased = SNR_prior_phase_based;
                    TableRow.OutSNR_PostPhaseBased = SNR_posterior_phase_based;
                    TableRow.OutSNR_PriorPhaseBasedFullCov = SNR_prior_phase_based_fullcov;
                    TableRow.OutSNR_PostPhaseBasedFullCov = SNR_posterior_phase_based_fullcov;
                    TableRow.OutSNR_PriorTimeBased = SNR_prior_time_based;
                    TableRow.OutSNR_PostTimeBased = SNR_posterior_time_based;

                    % Append results
                    if OneRowSet == false
                        ResultsTable = TableRow;
                        OneRowSet = true;
                    else
                        ResultsTable = cat(1, ResultsTable, TableRow);
                    end

                    if 0
                        t = (0 : length(x) - 1)/fs;
                        figure;
                        plot(t, x);
                        hold on;
                        plot(t(peaks_indexes), sig(peaks_indexes),'ro');
                        grid
                        xlabel('time (s)');
                        ylabel('Amplitude');
                        title('Noisy ECG and the detected R-peaks');
                        set(gca, 'fontsize', 16)
                    end

                    if 1
                        lg = {};
                        t = (0 : length(x) - 1)/fs;
                        figure
                        plot(t, x); lg = cat(2, lg, 'Noisy ECG');
                        hold on
                        plot(t(peaks_indexes), sig(peaks_indexes),'ro', 'markersize', 12); lg = cat(2, lg, 'R-peaks');
                        % plot(t, data_prior_est_time_based, 'linewidth', 2); lg = cat(2, lg, 'ECG prior estimate (time based)');
                        % plot(t, data_posterior_est_time_based, 'linewidth', 2); lg = cat(2, lg, 'ECG posterior estimate (time based)');
                        plot(t, data_prior_est_phase_based, 'linewidth', 2); lg = cat(2, lg, 'ECG prior estimate (phase based)');
                        plot(t, data_posterior_est_phase_based, 'linewidth', 2); lg = cat(2, lg, 'ECG posterior estimate (phase based)');
                        plot(t, data_prior_est_phase_based_fullcov, 'linewidth', 2); lg = cat(2, lg, 'ECG prior estimate (phase based full-cov.)');
                        plot(t, data_posterior_est_phase_based_fullcov, 'linewidth', 2); lg = cat(2, lg, 'ECG posterior estimate (phase based full-cov.)');
                        plot(t, sig, 'linewidth', 2); lg = cat(2, lg, 'Original ECG');
                        grid
                        legend(lg)
                        xlabel('time (s)');
                        ylabel('Amplitude');
                        title('Filtering results');
                        set(gca, 'fontsize', 16);
                    end

                    disp('Filtering performance:');
                    disp([' Input SNR (desired) = ' num2str(SNR_pre_set), 'dB']);
                    disp([' Input SNR (actual) = ' num2str(SNR_pre), 'dB']);
                    disp([' Output SNR (prior phase-based) = ' num2str(SNR_prior_phase_based), 'dB']);
                    disp([' Output SNR (prior phase-based full-cov) = ' num2str(SNR_prior_phase_based_fullcov), 'dB']);
                    disp([' Output SNR (prior time-based) = ' num2str(SNR_prior_time_based), 'dB']);
                    disp([' Output SNR (posterior phase-based) = ' num2str(SNR_posterior_phase_based), 'dB']);
                    disp([' Output SNR (posterior phase-based full-cov) = ' num2str(SNR_posterior_phase_based_fullcov), 'dB']);
                    disp([' Output SNR (posterior time-based) = ' num2str(SNR_posterior_time_based), 'dB']);
                    writetable(ResultsTable, ofname)
                end
            end
        end
    end
end


% if 0
%     hh = hamming(35);
%     % peaks = PeakDetection(data(1, :), 1.0/fs);
%     peaks = zeros(1, length(data(1, :)));
%     peak_points = 1 : 193 : length(data(1, :));
%     peaks(peak_points) = 1 + 0.05*rand(1, length(peak_points));
%     data = filter(hh - hh(end), 1, peaks);
%     % data = data + 0.1*std(data)*randn(size(data));
% end


