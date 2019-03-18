import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import os
#np.set_printoptions(threshold=np.nan)
import soundfile as sf


#  transfer wav data to three features and store as npy format
def process_wav(wav_path):
    y, osr = sf.read(wav_path, subtype='PCM_16', channels=1, samplerate=48000,
                    endian='LITTLE') #, start=56640, stop=262560)

    sr = 32000
    y = librosa.resample(y, osr, sr)
    #  filter zero
    # for index, v in enumerate(y):
    #     if v != 0:
    #         y = y[index:]
    #         break
    # end = len(y)
    # for i in reversed(y):
    #     if i != 0:
    #         y = y[:end]
    #     else:
    #         end -= 1


    #使用DIO算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=800.0, channels_in_octave=2, frame_period=pw.default_frame_period)
    print(_f0.shape)

    #使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)
    
    code_sp = pw.code_spectral_envelope(_sp, sr, 60)
    print(_sp.shape, code_sp.shape)
    #计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    code_ap = pw.code_aperiodicity(_ap, sr)
    print(_ap.shape, code_ap.shape)

    return _f0, code_sp, code_ap

def process_phon_label(label_path):
    file = open(label_path, 'r')

    phon_list = []
    all_phon = []
    try:
        text_lines = file.readlines()
        print(type(text_lines), text_lines)
        for line in text_lines:
            line = line.replace('\n', '')
            l_c = line.split(' ')
            phn = l_c[2]
            tup = (float(l_c[0])*200/10000000, float(l_c[1])*200/10000000, phn)
            print(tup)
            phon_list.append(tup)
            if phn not in all_phon:
                all_phon.append(phn)
    finally:
        file.close()

    all_phon.append('none')
    return phon_list, all_phon

if __name__ == '__main__':

    wav_path = './raw/nitech_jp_song070_f001_003.raw'
    f0, sp, ap = process_wav(wav_path)
    v_uv = f0 > 0

    spmin = -21.62037003104595
    spmax = 5.839000361601009
    sp = (sp-spmin)/(spmax-spmin)
    np.save('prepared_data/v_uv.npy', f0)
    np.save('prepared_data/f0.npy', f0)
    np.save('prepared_data/sp.npy', sp)
    np.save('prepared_data/ap.npy', ap)

    f0_coarse = np.rint(f0*1024/np.max(f0)).astype(np.int)
    print(np.max(f0_coarse))

    txt_path = './raw/nitech_jp_song070_f001_003_mono.lab'
    phon_list, all_phon = process_phon_label(txt_path)

    label_list = []
    oh_list = []
    for i in range(len(f0)):
        pre_phn, cur_phn, next_phn, pos_in_phon = (0, 0, 0, 0)
        for j in range(len(phon_list)):
            if phon_list[j][0] <= i <= phon_list[j][1]:
                cur_phn = all_phon.index(phon_list[j][2])
                if j == 0:
                    pre_phn = all_phon.index('none')
                else:
                    pre_phn = all_phon.index(phon_list[j-1][2])

                if j == len(phon_list)-1:
                    next_phn = all_phon.index('none')
                else:
                    next_phn = all_phon.index(phon_list[j+1][2])


                width = phon_list[j][1] - phon_list[j][0] + 1
                if i+1 <= width/3:
                    pos_in_phon = 0
                elif width/3 < i+1 <= 2*width/3:
                    pos_in_phon = 1
                else:
                    pos_in_phon = 2

        label_list.append([pre_phn, cur_phn, next_phn, pos_in_phon, f0_coarse[i]])

        # onehot
        pre_phn_oh = np.zeros(len(all_phon))
        cur_phn_oh = np.zeros(len(all_phon))
        next_phn_oh = np.zeros(len(all_phon))
        pos_in_phon_oh = np.zeros(3)
        f0_coarse_oh = np.zeros(np.max(f0_coarse)+1)

        pre_phn_oh[pre_phn] = 1
        cur_phn_oh[cur_phn] = 1
        next_phn_oh[next_phn] = 1
        pos_in_phon_oh[pos_in_phon] = 1
        f0_coarse_oh[f0_coarse[i]] = 1

        oh_list.append(np.concatenate((pre_phn_oh, cur_phn_oh, next_phn_oh, pos_in_phon_oh, f0_coarse_oh)).astype(np.int8))
        print(len(oh_list[-1]), np.sum(oh_list[-1]))

    np.save('prepared_data/condition.npy', oh_list)