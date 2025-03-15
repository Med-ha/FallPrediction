import matplotlib.pyplot as plt

def plot_eeg(signal, label, sampling_rate, title):
    """
    Plots the given EEG or ACC signal over time.

    :param signal: The signal data (numpy array or list).
    :param label: The label of the signal (e.g., 'R6', 'x_dir').
    :param sampling_rate: Sampling rate of the signal (in Hz).
    :param title: Title of the plot.
    """
    time_axis = [i / sampling_rate for i in range(len(signal))]  # Convert samples to time in seconds

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Example usage
plot_eeg(EEG_whole['R6'], 'R6', 200, 'EEG over Time')
plot_eeg(ACC_whole_x['x_dir'], 'x_dir', 200, 'ACC over Time')
