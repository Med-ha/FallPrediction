import numpy as np
import scipy.signal as signal
import mne  # For EEG file handling

class Preprocessing:
    @staticmethod
    def get_edf_signal(edf_path, channel):
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        return raw.get_data(picks=channel).flatten()

    @staticmethod
    def normalize_eeg(eeg_signal):
        return (eeg_signal - np.mean(eeg_signal[:40])) / np.max(eeg_signal)

    @staticmethod
    def extract_features(eeg_signal, order=3):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=order)
        return pca.fit_transform(eeg_signal.reshape(-1, 1)).flatten()
