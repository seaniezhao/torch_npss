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


process_wav('/home/sean/pythonProj/torch_npss/data/raw/nitech_jp_song070_f001_055.raw')
# wav_path = './data/gen_samples/gen.npy'
# code_sp = np.load(wav_path).astype(np.double)
#
# wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/train/sp/nitech_jp_song070_f001_014_sp.npy'
# code_sp1 = np.load(wav_path).astype(np.double)
#
# spmin = -21.62037003104595
# spmax = 5.839000361601009
# code_sp = code_sp * (spmax-spmin) + spmin
# code_sp1 = code_sp1 * (spmax-spmin) + spmin
#
# sp = pw.decode_spectral_envelope(code_sp, 32000, 1024)
# sp1 = pw.decode_spectral_envelope(code_sp1, 32000, 1024)
# #print(data)
#
#
# #print(ap)
#
#
#
# plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom',
#                        interpolation='none')
# plt.show()
# plt.imshow(np.log(np.transpose(sp1)), aspect='auto', origin='bottom',
#                        interpolation='none')
# plt.show()
#
# f0 = np.load('./data/prepared_data/f0.npy').astype(np.double)
# ap = np.load('./data/prepared_data/ap.npy').astype(np.double)
#
# ap = pw.decode_aperiodicity(ap, 32000, 1024)
# # 合成原始语音
# synthesized = pw.synthesize(f0, sp1, ap, 32000, pw.default_frame_period)
# # 1.输出原始语音
# sf.write('./data/gen_wav/synthesized.wav', synthesized, 32000)