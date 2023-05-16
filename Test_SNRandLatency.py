import unittest
import numpy as np
from numpy import random
import timeit
import matplotlib.pyplot as plt
import passKeyInfoToSolidity

'''
测试框架在不同SNR下的延迟

需要输入：
1. 对象数组（传参）
2. 不同信噪比（自定义）
3. 时延（传参）
'''
#signal = np.array([1, 2, 3, 4, 5])
signal = passKeyInfoToSolidity.pass_parms_array
noise_levels = [0.1,4,10] # different SNR levels to test
delays = []

noise = None
def passNoise(noise):
    return noise

def test_delay():
    for noise_level in noise_levels:
        noise = random.normal(0, noise_level, signal.shape)
        noisy_signal = signal + noise

        start = timeit.default_timer()
        # do some operation on the noisy signal here
        end = timeit.default_timer()

        delay = end - start
        delays.append(passKeyInfoToSolidity.delay)

        print("SNR:", 20 * np.log10(np.linalg.norm(signal) / np.linalg.norm(noise)), "dB Delay:", delay,
                  "seconds")

def tearDown():
    plt.bar(noise_levels, delays)
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Delay (seconds)')
    plt.show()


if __name__ == '__main__':
    test_delay()
    tearDown()
