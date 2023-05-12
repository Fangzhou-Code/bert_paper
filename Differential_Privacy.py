import numpy as np

'''
添加噪声函数:
我们定义了一个名为add_noise的函数，它接受两个参数：原始数据和隐私预算（即ϵ）。
该函数首先计算灵敏度，然后根据隐私预算计算噪声范围，并使用Numpy的uniform函数生成均匀分布的噪声。
最后，将噪声添加到原始数据中得到带有差分隐私保护的数据。

差分隐私是一种广泛使用的隐私保护技术，但是在使用差分隐私的过程中，可能会出现如您所述的问题，即隐私保护后的数据与原始数据之间存在较大的差异性。这种情况下，可以考虑以下几种方法来解决问题：
    调整隐私参数：差分隐私的隐私参数ε越小，隐私保护程度越高，但噪音也会增加，从而导致还原后的数据与原数据差别更大。因此，可以适当调整隐私参数，以平衡隐私保护和数据还原准确性之间的关系。
    使用更好的噪声模型：在差分隐私中，添加噪声是为了保护隐私。如果噪声模型不够好，会导致添加的噪声过大或过小，从而影响数据还原的准确性。因此，选择更好的噪声模型，可以有效地缓解这个问题。
    数据增强：数据增强是指通过某些技术手段，增加原始数据的数量和多样性，从而提高还原后数据与原数据的相似性。例如，可以使用生成对抗网络（GAN）来生成更多样化的数据，或者使用数据扰动和数据脱敏等技术来增加原始数据的数量和多样性。
    算法优化：在差分隐私还原算法中，可能存在一些可以优化的地方，例如使用更高效的算法、优化算法参数、或者改进算法流程等。这些优化措施有助于提高还原后数据的准确性。
    综上所述，解决差分隐私还原后数据与原数据差别大的问题，需要综合考虑隐私保护和数据还原准确性之间的关系，选择适当的差分隐私方案，并结合具体情况采取相应的方法进行调整和优化。
'''
def add_noise(data, epsilon):
    # 计算灵敏度
    sensitivity = 1.0
    # 计算噪声范围
    noise_range = sensitivity / epsilon
    # 添加噪声

    noisy_data = data + np.random.uniform(-noise_range, noise_range, size=data.shape)

    # 计算信噪比
    # snr = 20 * np.log10(np.linalg.norm(data) / np.linalg.norm(noisy_data))
    # print("Signal-to-Noise Ratio (SNR):", snr, "dB")

    print("=====Differential_Privacy=====")
    print("原始数据：", data)
    print("添加噪声后的数据：", noisy_data)
    print("=====end=====")
    return noisy_data

def remove_noise(noisy_data, epsilon):
    # 还原数据
    restored_data = noisy_data - add_noise(np.zeros_like(noisy_data), epsilon)
    print("=====Differential_Privacy=====")
    print("还原后的数据：", restored_data)
    print("=====end=====")
    return restored_data

if __name__ == '__main__':
    # 数据准备
    data = np.array([1, 2, 3, 4, 5])
    epsilon = 0.9
    # 添加噪声
    noisy_data = add_noise(data, epsilon)
    # 输出结果
    print("原始数据：", data)
    print("添加噪声后的数据：", noisy_data)

    # 还原数据
    restored_data = remove_noise(noisy_data, epsilon)
    # 输出结果
    print("还原后的数据：", restored_data)