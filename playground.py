import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import os
#np.set_printoptions(threshold=np.nan)

wav_path = './data/1 (1, 2, 3, 4).wav'


y, sr = librosa.load(wav_path, dtype=np.float64)
#print(data)

f0, sp, ap = pw.wav2world(y, sr)

#分布提取参数
#使用DIO算法计算音频的基频F0
_f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=600.0, channels_in_octave=2, frame_period=pw.default_frame_period)
print(_f0.shape)

# 使用CheapTrick算法计算音频的频谱包络
_sp = pw.cheaptrick(y, _f0, t, sr)

code_sp = pw.code_spectral_envelope(_sp, sr, 80)
print(_sp.shape, code_sp.shape)
# 计算aperiodic参数
_ap = pw.d4c(y, _f0, t, sr)

code_ap = pw.code_aperiodicity(_ap, sr)
print(_ap.shape, code_ap.shape)


#print(ap)

plt.plot(_f0)
plt.show()

plt.imshow(np.log(code_sp), cmap='gray')
plt.show()
plt.imshow(_ap, cmap='gray')
plt.show()