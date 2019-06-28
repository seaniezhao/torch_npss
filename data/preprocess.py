import librosa
import pyworld as pw
import numpy as np
import os
import soundfile as sf
import fnmatch
from data_util import code_harmonic, decode_harmonic

sp_min = 0
sp_max = 0

ap_min = 0
ap_max = 0

rt_folder = 'timbre_model'
tr_folder = 'timbre_model/train'

sp_folder = 'timbre_model/train/sp/'
ap_folder = 'timbre_model/train/ap/'
vuv_folder = 'timbre_model/train/vuv/'
condition_folder = 'timbre_model/train/condition/'

te_folder = 'timbre_model/test'

te_sp_folder = 'timbre_model/test/sp/'
te_ap_folder = 'timbre_model/test/ap/'
te_vuv_folder = 'timbre_model/test/vuv/'
te_condition_folder = 'timbre_model/test/condition/'

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
#  transfer wav data to three features and store as npy format
def process_wav(wav_path):
    y, osr = sf.read(wav_path, subtype='PCM_16', channels=1, samplerate=48000,
                    endian='LITTLE') #, start=56640, stop=262560)

    sr = 32000
    if osr != sr:
        y = librosa.resample(y, osr, sr)

    #使用harvest算法计算音频的基频F0
    _f0, t = pw.harvest(y, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=pw.default_frame_period)
    _f0 = pw.stonemask(y, _f0, t, sr)
    print(_f0.shape)

    #使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)

    code_sp = code_harmonic(_sp, 60)
    print(_sp.shape, code_sp.shape)
    #计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    code_ap = pw.code_aperiodicity(_ap, sr)
    print(_ap.shape, code_ap.shape)

    return _f0, _sp, code_sp, _ap, code_ap


def process_phon_label(label_path):
    file = open(label_path, 'r')

    time_phon_list = []
    phon_list = []
    try:
        text_lines = file.readlines()
        print(type(text_lines), text_lines)
        for line in text_lines:
            line = line.replace('\n', '')
            l_c = line.split(' ')
            phn = l_c[2]
            tup = (float(l_c[0])*200/10000000, float(l_c[1])*200/10000000, phn)
            print(tup)
            time_phon_list.append(tup)
            if phn not in phon_list:
                phon_list.append(phn)
    finally:
        file.close()

    return time_phon_list, phon_list


def process_timbre_model_condition(time_phon_list, all_phon, f0):

    f0_coarse = np.rint(f0*(f0_bin-1)/f0_max).astype(np.int)
    f0_coarse[f0_coarse>255] = 255
    print('Max f0', np.max(f0_coarse))

    label_list = []
    oh_list = []
    for i in range(len(f0)):
        pre_phn, cur_phn, next_phn, pos_in_phon = (0, 0, 0, 0)
        for j in range(len(time_phon_list)):
            if time_phon_list[j][0] <= i <= time_phon_list[j][1]:
                cur_phn = all_phon.index(time_phon_list[j][2])
                if j == 0:
                    pre_phn = all_phon.index('none')
                else:
                    pre_phn = all_phon.index(time_phon_list[j - 1][2])

                if j == len(time_phon_list) - 1:
                    next_phn = all_phon.index('none')
                else:
                    next_phn = all_phon.index(time_phon_list[j + 1][2])

                begin = time_phon_list[j][0]
                end = time_phon_list[j][1]
                width = end - begin



                if i - begin < width / 3:
                    pos_in_phon = 0
                elif width / 3 <= i - begin < 2 * width / 3:
                    pos_in_phon = 1
                else:
                    pos_in_phon = 2

        label_list.append([pre_phn, cur_phn, next_phn, pos_in_phon, f0_coarse[i]])

        # onehot
        pre_phn_oh = np.zeros(len(all_phon))
        cur_phn_oh = np.zeros(len(all_phon))
        next_phn_oh = np.zeros(len(all_phon))
        pos_in_phon_oh = np.zeros(3)
        f0_coarse_oh = np.zeros(f0_bin)

        pre_phn_oh[pre_phn] = 1
        cur_phn_oh[cur_phn] = 1
        next_phn_oh[next_phn] = 1
        pos_in_phon_oh[pos_in_phon] = 1
        f0_coarse_oh[f0_coarse[i]] = 1

        oh_list.append(
            np.concatenate((pre_phn_oh, cur_phn_oh, next_phn_oh, pos_in_phon_oh, f0_coarse_oh)).astype(np.int8))
        if i == len(f0) - 1:
            print(len(oh_list[-1]), np.sum(oh_list[-1]))

    return oh_list

if __name__ == '__main__':
    if not os.path.exists(rt_folder):
        os.mkdir(rt_folder)
    # train folders
    if not os.path.exists(tr_folder):
        os.mkdir(tr_folder)

    if not os.path.exists(sp_folder):
        os.mkdir(sp_folder)
    if not os.path.exists(ap_folder):
        os.mkdir(ap_folder)
    if not os.path.exists(vuv_folder):
        os.mkdir(vuv_folder)
    if not os.path.exists(condition_folder):
        os.mkdir(condition_folder)
    # test folders
    if not os.path.exists(te_folder):
        os.mkdir(te_folder)

    if not os.path.exists(te_sp_folder):
        os.mkdir(te_sp_folder)
    if not os.path.exists(te_ap_folder):
        os.mkdir(te_ap_folder)
    if not os.path.exists(te_vuv_folder):
        os.mkdir(te_vuv_folder)
    if not os.path.exists(te_condition_folder):
        os.mkdir(te_condition_folder)

    test_names = ['nitech_jp_song070_f001_015', 'nitech_jp_song070_f001_029', 'nitech_jp_song070_f001_040']

    raw_folder = './raw_one'

    all_phon = ['none']
    data_to_save = []

    supportedExtensions = '*.raw'
    for dirpath, dirs, files in os.walk(raw_folder):
        for file in fnmatch.filter(files, supportedExtensions):
            file_name = file.replace('.raw','')
            raw_path = os.path.join(dirpath, file)
            txt_path = raw_path.replace('.raw', '.lab')
            f0, _sp, code_sp, _ap, code_ap = process_wav(raw_path)
            v_uv = f0 > 0

            time_phon_list, phon_list = process_phon_label(txt_path)
            for item in phon_list:
                if item not in all_phon:
                    all_phon.append(item)

            data_to_save.append((file_name, time_phon_list, f0, code_sp, code_ap, v_uv))

            _sp_min = np.min(code_sp)
            _sp_max = np.max(code_sp)
            if _sp_min < sp_min:
                sp_min = _sp_min
            if _sp_max > sp_max:
                sp_max = _sp_max

            _ap_min = np.min(code_ap)
            _ap_max = np.max(code_ap)
            if _ap_min < ap_min:
                ap_min = _ap_min
            if _ap_max > ap_max:
                ap_max = _ap_max


    np.save('timbre_model/min_max_record.npy', [sp_min, sp_max, ap_min, ap_max])
    np.save('timbre_model/all_phonetic.npy', all_phon)


    for file_name, time_phon_list, f0, code_sp, code_ap, v_uv in data_to_save:
        oh_list = process_timbre_model_condition(time_phon_list, all_phon, f0)

        code_sp = (code_sp - sp_min) / (sp_max - sp_min) - 0.5
        code_ap = (code_ap - ap_min) / (ap_max - ap_min) - 0.5

        test = False
        for n in test_names:
            if n in file_name:
                test = True
        if test:
            np.save(te_condition_folder + file_name + '_condi.npy', oh_list)
            np.save(te_sp_folder + file_name + '_sp.npy', code_sp)
            np.save(te_ap_folder + file_name + '_ap.npy', code_ap)
            np.save(te_vuv_folder + file_name + '_vuv.npy', v_uv)
        else:
            np.save(condition_folder + file_name + '_condi.npy', oh_list)
            np.save(sp_folder + file_name + '_sp.npy', code_sp)
            np.save(ap_folder + file_name + '_ap.npy', code_ap)
            np.save(vuv_folder + file_name + '_vuv.npy', v_uv)

        # np.save('prepared_data/f0.npy', f0)


