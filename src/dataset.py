import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d


class OpticalAPSKDataset(Dataset):
    def __init__(self, num_samples=25000, symbol_len=24):
        self.num_samples = num_samples
        self.symbol_len = symbol_len

        self.amps = [3.0, 5.0, 7.0, 9.0]
        self.phases = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]

        self.classes = []
        for a_idx in range(4):
            for p_idx in range(4):
                self.classes.append((self.amps[a_idx], self.phases[p_idx]))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = np.random.randint(0, 16)
        amp, phase = self.classes[label]

        t = np.linspace(0, 2.0 * np.pi, self.symbol_len, endpoint=False)
        clean_wave = amp * np.sin(t + phase)

        # Baseline & DC Kayması (DC filtrelenebilir)
        dc_offset = np.random.uniform(-1.0, 1.0)
        slope = np.random.uniform(-0.05, 0.05)
        baseline = dc_offset + slope * np.arange(self.symbol_len)

        # Optik Bulanıklık
        sigma = np.random.uniform(0.2, 0.8)
        blurred_wave = gaussian_filter1d(clean_wave, sigma=sigma)

        # Gürültü (AWGN)
        noise = np.random.normal(0.0, 0.35, size=self.symbol_len)

        distorted_signal = blurred_wave + baseline + noise

        # DC Bileşeni Kaldır (Sadece Ortalama Çıkarılır, Genlik / Std Korunur!)
        distorted_signal = distorted_signal - np.mean(distorted_signal)

        # Sabit Dinamik Aralık Ölçekleme (Genlik bilgisini ezmeden [-1, +1] bandına yaklaştır)
        distorted_signal = distorted_signal / 12.0

        x_tensor = torch.tensor(distorted_signal, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(label, dtype=torch.long)

        return x_tensor, y_tensor