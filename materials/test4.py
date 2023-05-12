import unittest
import numpy as np
from numpy import random
import timeit
import matplotlib.pyplot as plt


class TestDelay(unittest.TestCase):
    def setUp(self):
        self.signal = np.array([1, 2, 3, 4, 5])
        self.noise_levels = [0.1, 1, 10]  # different SNR levels to test
        self.delays = []

    def test_delay(self):
        for noise_level in self.noise_levels:
            noise = random.normal(0, noise_level, self.signal.shape)
            noisy_signal = self.signal + noise

            start = timeit.default_timer()
            # do some operation on the noisy signal here
            end = timeit.default_timer()

            delay = end - start
            self.delays.append(delay)

            print("SNR:", 20 * np.log10(np.linalg.norm(self.signal) / np.linalg.norm(noise)), "dB Delay:", delay,
                  "seconds")

    def tearDown(self):
        plt.bar(self.noise_levels, self.delays)
        plt.xlabel('Signal-to-Noise Ratio (SNR)')
        plt.ylabel('Delay (seconds)')
        plt.show()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
