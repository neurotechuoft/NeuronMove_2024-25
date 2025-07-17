channels_to_plot = [1, 2, 3, 4, 5]; % Example: Fp1, Fpz, Fp2, AF7, AF3
time_window_seconds = 5;
num_samples_to_plot = EEG.srate * time_window_seconds; % Number of data points for 5 seconds
time_vector = (0:num_samples_to_plot-1) / EEG.srate;
figure; % Opens a new figure window
hold on; % Keeps multiple plots on the same axes

for i = 1:length(channels_to_plot)
    chan_idx = channels_to_plot(i);
    % Offset each channel vertically for easier viewing
    plot(time_vector, EEG.data(chan_idx, 1:num_samples_to_plot) + (i * 50)); % + (i * offset) for separation
    text(time_vector(end)+0.1, EEG.data(chan_idx, num_samples_to_plot) + (i * 50), EEG.chanlocs(chan_idx).labels); % Label channel
end

hold off;
xlabel('Time (s)');
ylabel('Amplitude (\muV, offset for clarity)');
title(['EEG Data for Channels ', num2str(channels_to_plot), ' (First ', num2str(time_window_seconds), ' seconds)']);
grid on;