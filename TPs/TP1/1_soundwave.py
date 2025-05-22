import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import gaussian

class Signal:
    def show_sinus(self, sampleRate: int = 44100, freq: int = 440, duration: int = 1):
        t = np.linspace(0, duration, sampleRate * duration)
        y = np.sin(2 * np.pi * freq * t)

        faded_in_signal = self.fade_in(y, sampleRate, duration)
        # faded_out_signal = self.fade_out(y, sampleRate, duration)
        noised_signal = self.apply_noise(y, sampleRate, duration)
        # smoothed_signal = self.smoothing(noised_signal)
        fig, axs = plt.subplots(3)
        axs[0].plot(t[:1000], y[:1000])
        axs[0].set_title("Sinus LA")
        axs[1].plot(t[:1000], faded_in_signal[:1000], 'tab:orange')
        axs[1].set_title("Sinus LA fade-in")
        axs[2].plot(t[:1000], noised_signal[:1000], 'tab:green')
        axs[2].set_title("Sinus LA noised")
        plt.show()


        # self.plot_signal("Sinus LA", t, y, 1000, 1000)
        # self.plot_signal("Sinus LA fade-in", t, faded_in_signal, 1000, 1000)
        # self.plot_signal("Sinus LA fade-out", t, faded_out_signal)
        # self.plot_signal("Sinus LA noised", t, noised_signal, 1000, 1000)
        # self.plot_signal("Sinus LA smoothed", t, smoothed_signal, 1000, 1000)

    def fade_in(self, signal: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
        samples = int(sample_rate * duration)
        fade_curve = np.linspace(0, 1, samples)
        faded_signal = signal.copy()
        faded_signal[:samples] *= fade_curve
        return faded_signal

    def fade_out(self, signal: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
        samples = int(sample_rate * duration)
        fade_curve = np.linspace(1, 0, samples)
        faded_signal = signal.copy()
        faded_signal[-samples:] *= fade_curve
        return faded_signal

    def apply_noise(self, signal: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
        samples = int(sample_rate * duration)
        noise_curve = np.random.normal(0, 10, sample_rate * duration)
        noised_signal = signal.copy()
        noised_signal[:samples] += noise_curve
        return noised_signal

    def smoothing(self, signal: np.ndarray, window_size: int = 50) -> np.ndarray:
        kernel = gaussian(window_size, std=window_size / 6)
        kernel /= kernel.sum()
        smoothed_curve = np.convolve(signal, kernel, mode="same")
        return smoothed_curve

    def plot_signal(self, title, x, y, x_interval=None, y_interval=None):
        if x_interval is not None and y_interval is not None:
            plt.plot(x[:x_interval], y[:y_interval])
        else:
            plt.plot(x, y)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    signal = Signal()
    signal.show_sinus()
