import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fft

if __name__ == '__main__':
    rate, data = wavfile.read('../res/sound.wav')
    samples_num = len(data)
    scaled_data = data / (2. ** 15)
    time_arr = np.arange(0, float(samples_num), 1) / rate
    plt.rcParams['figure.figsize'] = [20, 5]
    plt.plot(time_arr, scaled_data, linewidth=0.3, alpha=0.7, color='#004bc6')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig('../res/sound.png')
    plt.show()

    T = 1.0 / samples_num
    yf = fft.fft(scaled_data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), samples_num // 2)
    plt.plot(xf, (np.abs(yf[0:samples_num // 2])))
    plt.grid()
    plt.savefig('../res/sound_after_fft.png')
    plt.show()

    for i in range(len(xf)):
        if 15000 < xf[i] < 30000:
            yf[i] = 0
    plt.plot(xf, (np.abs(yf[0:samples_num // 2])))
    plt.grid()
    plt.savefig('../res/sound_after_remove.png')
    plt.show()

    y = np.real(fft.ifft(yf))
    wavfile.write("../res/sound_after_fft.wav", len(y), y)

    plt.plot(time_arr, y)
    plt.grid()
    plt.savefig('../res/sound_after.png')
    plt.show()
