import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import os
#np.set_printoptions(threshold=np.nan)


#  transfer wav data to three features and store as npy format
def process_wav(wav_path):
    y, sr = librosa.load(wav_path, dtype=np.float64)

    #使用DIO算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=600.0, channels_in_octave=2, frame_period=pw.default_frame_period)
    print(_f0.shape)

    #使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)
    
    code_sp = pw.code_spectral_envelope(_sp, sr, 80)
    print(_sp.shape, code_sp.shape)
    #计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    code_ap = pw.code_aperiodicity(_ap, sr)
    print(_ap.shape, code_ap.shape)

    return _f0, code_sp, code_ap

if __name__ == '__main__':

    wav_path = './Stereo Out.wav'
    f0, sp, ap = process_wav(wav_path)
    np.save('prepared_data/f0.npy', f0)
    np.save('prepared_data/sp.npy', sp)
    np.save('prepared_data/ap.npy', ap)
