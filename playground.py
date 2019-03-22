import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
#np.set_printoptions(threshold=np.nan)


def process_wav(wav_path):
    y, osr = sf.read(wav_path, subtype='PCM_16', channels=1, samplerate=48000,
                     endian='LITTLE')  # , start=56640, stop=262560)

    sr = 32000
    y = librosa.resample(y, osr, sr)

    # 使用DIO算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=800.0, channels_in_octave=2, frame_period=pw.default_frame_period)
    print(_f0.shape)

    # 使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)

    code_sp = pw.code_spectral_envelope(_sp, sr, 60)
    print(_sp.shape, code_sp.shape)
    # 计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    code_ap = pw.code_aperiodicity(_ap, sr)
    print(_ap.shape, code_ap.shape)

    np.save('data/prepared_data/f0', _f0)
    np.save('data/prepared_data/ap', code_ap)

    # 合成原始语音
    synthesized = pw.synthesize(_f0-200, _sp, _ap, 32000, pw.default_frame_period)
    # 1.输出原始语音
    sf.write('./data/gen_wav/test-200.wav', synthesized, 32000)

#process_wav('/home/sean/pythonProj/torch_npss/data/raw/nitech_jp_song070_f001_055.raw')


def get_feature(wav_path):
    y, osr = sf.read(wav_path)  # , start=56640, stop=262560)

    sr = 32000
    y = librosa.resample(y, osr, sr)

    # 使用DIO算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=800.0, channels_in_octave=2, frame_period=pw.default_frame_period)
    print(_f0.shape)

    # 使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)

    # 计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    return _f0, _sp, _ap

a = '/home/sean/Desktop/f0_tets/counddown_ori.wav'
b = '/home/sean/Desktop/f0_tets/counddown_joe.wav'

af0, asp, aap = get_feature(a)
bf0, bsp, bap = get_feature(b)

plt.plot(af0)
plt.show()
plt.plot(bf0)
plt.show()

bf0[18:1019] = (bf0[18:1019] > 0)*af0
#
# for i,f0 in enumerate(bf0):
#     if i>1 and f0 == 0:
#         bf0[i] = bf0[i-1]



# 合成原始语音
synthesized = pw.synthesize(bf0[18:1019]/2.5, bsp[18:1019], bap[18:1019], 32000, pw.default_frame_period)
# 1.输出原始语音
sf.write('./data/gen_wav/countdown.wav', synthesized, 32000)
