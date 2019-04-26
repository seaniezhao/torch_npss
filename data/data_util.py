import scipy.fftpack as fftpack
import librosa
import pyworld as pw
import numpy as np
import os
import soundfile as sf
import fnmatch
import matplotlib.pyplot as plt
import pysptk
from librosa.display import specshow
import copy


gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.35
en_floor = 10 ** (-80 / 20)


def code_harmonic(sp, order):

    #get mcep
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    #do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real

    return mfsc


def decode_harmonic(mfsc, fftlen):
    # get mcep back
    mceps_mirror = np.fft.irfft(mfsc)
    mceps_back = mceps_mirror[:, :60]
    mceps_back[:, 0] /= 2
    mceps_back[:, -1] /= 2

    #get sp
    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, mceps, alpha, gamma, fftlen=fftlen).real)

    return spSm


if __name__ == '__main__':
    y, osr = sf.read('cut_raw/nitech_jp_song070_f001_004_1.raw', subtype='PCM_16', channels=1, samplerate=48000,
                     endian='LITTLE')  # , start=56640, stop=262560)

    sr = 32000
    if osr != sr:
        y = librosa.resample(y, osr, sr)

    # D = np.abs(librosa.stft(y, hop_length=160)) ** 2
    # #D_db = librosa.power_to_db(D, ref=np.max)
    # S = librosa.feature.melspectrogram(S=D)
    # ptd_S = librosa.power_to_db(S)
    # mfcc = librosa.feature.mfcc(S=ptd_S, n_mfcc=60)
    #
    #
    # 使用DIO算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=800.0, channels_in_octave=2, frame_period=pw.default_frame_period)
    print(_f0.shape)

    # 使用CheapTrick算法计算音频的频谱包络
    sp = pw.cheaptrick(y, _f0, t, sr)

    _ap = pw.d4c(y, _f0, t, sr)
    # ptd_S = librosa.power_to_db(np.transpose(_sp))
    # tran_ptd_S = (ptd_S - 0.45)/(1 - 0.45*ptd_S)
    # mfcc = librosa.feature.mfcc(S=tran_ptd_S, n_mfcc=60)
    #
    # _sp_min = np.min(mfcc)
    # _sp_max = np.max(mfcc)
    # mfcc = (mfcc - _sp_min)/(_sp_max - _sp_min)
    #
    # code_sp = pw.code_spectral_envelope(_sp, sr, 60)
    # t_code_sp = np.transpose(code_sp)
    #
    # _sp_min = np.min(t_code_sp)
    # _sp_max = np.max(t_code_sp)
    # t_code_sp = (t_code_sp - _sp_min) / (_sp_max - _sp_min)
    #
    # plt.imshow(mfcc, aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    # plt.imshow(t_code_sp, aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    #
    # decode_sp = pw.decode_spectral_envelope(code_sp, 32000, 2048)
    # x = code_harmonic(_sp)
    order = 60
    gamma = 0
    mcepInput = 3  # 0 for dB, 3 for magnitude
    alpha = 0.35
    fftlen = (sp.shape[1] - 1) * 2
    en_floor = 10 ** (-80 / 20)

    # Reduction and Interpolation
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    # scale_mceps = copy.copy(mceps)
    # scale_mceps[:, 0] *=2
    # scale_mceps[:, -1] *=2
    # mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    # mfsc = np.fft.rfft2(mirror).real
    mfsc = fftpack.dct(mceps, norm='ortho')

    specshow(mfsc.T, sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('MCEPS')
    plt.tight_layout()
    plt.show()

    # itest = np.fft.ifft2(mfsc).real
    itest = fftpack.idct(mfsc, norm='ortho')

    specshow(itest.T, sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('itest')
    plt.tight_layout()
    plt.show()

    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, itest, alpha, gamma, fftlen=fftlen).real)

    specshow(10 * np.log10(sp.T), sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('Original envelope spectrogram')
    plt.tight_layout()
    plt.show()

    specshow(10 * np.log10(spSm.T), sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('Smooth envelope spectrogram')
    plt.tight_layout()
    plt.show()

    synthesized = pw.synthesize(_f0, spSm, _ap, 32000, pw.default_frame_period)
    # 1.输出原始语音
    sf.write('gen_wav/fffff.wav', synthesized, 32000)

    # mgc = pysptk.mcep(np.sqrt(fft), 59, 0.35, itype=3)
    # mfsc = np.exp(pysptk.mgc2sp(mgc, 0.35, fftlen=2048).real)
    # pysptk.mgc2sp
    # pass
