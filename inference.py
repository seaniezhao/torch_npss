from model.wavenet_model import *
from data.dataset import NpssDataset
import hparams
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf




def load_latest_model_from(mtype, location):

    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)

    print("load model " + newest_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mtype == 0:
        hparam = hparams.create_harmonic_hparams()
    else:
        hparam = hparams.create_aperiodic_hparams()

    model = WaveNetModel(hparam, device).to(device)
    states = torch.load(newest_file)
    model.load_state_dict(states['state_dict'])

    return model


def load_timbre(path, m_type, mx, mn):
    load_t = np.load(path).astype(np.double)

    load_t = load_t * (mx - mn) + mn
    decode_sp = pw.decode_spectral_envelope(load_t, 32000, 1024)
    if m_type == 1:
        decode_sp = pw.decode_aperiodicity(load_t, 32000, 1024)

    return decode_sp


#  type 0:harmonic, 1:aperiodic, 2 vuv
def generate_timbre(m_type, mx, mn, condition, cat_input=None, init_input=None):
    model_path = 'snapshots/harmonic'
    if m_type == 1:
        model_path = 'snapshots/aperiodic'
    model = load_latest_model_from(m_type, model_path)
    sample = model.generate(condition, cat_input, init_input).transpose(0,1).cpu().numpy().astype(np.double)
    sample = sample * (mx - mn) + mn

    decode_sp = None
    if m_type == 0:
        decode_sp = pw.decode_spectral_envelope(sample, 32000, 1024)
    elif m_type == 1:
        decode_sp = pw.decode_aperiodicity(sample, 32000, 1024)

    return decode_sp


def get_cat_input():

    wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/test/sp/nitech_jp_song070_f001_003_sp.npy'

    code_sp = np.load(wav_path).astype(np.double)
    return torch.Tensor(code_sp).transpose(0, 1)


def get_first_input():

    wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/test/sp/nitech_jp_song070_f001_003_sp.npy'
    #wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/train/ap/nitech_jp_song070_f001_055_ap.npy'

    code_sp = np.load(wav_path).astype(np.double)
    return torch.Tensor(code_sp).transpose(0, 1)


def get_condition():

    c_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/test/condition/nitech_jp_song070_f001_003_condi.npy'
    conditon = np.load(c_path).astype(np.float)
    return torch.Tensor(conditon).transpose(0, 1)





if __name__ == '__main__':

    [sp_min, sp_max, ap_min, ap_max] = np.load('/home/sean/pythonProj/torch_npss/data/timbre_model/min_max_record.npy')
    condi = get_condition()
    cat_input = get_cat_input()
    fist_input = get_first_input()

    sp = generate_timbre(0, sp_max, sp_min, condi, None, fist_input)

    plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom', interpolation='none')
    plt.show()

    sp1 = load_timbre('/home/sean/pythonProj/torch_npss/data/timbre_model/test/sp/nitech_jp_song070_f001_003_sp.npy', 0, sp_max, sp_min)

    plt.imshow(np.log(np.transpose(sp1)), aspect='auto', origin='bottom', interpolation='none')
    plt.show()

    # ap = generate_timbre(1, ap_max, ap_min, condi, cat_input, fist_input)
    #
    # plt.imshow(np.log(np.transpose(ap)), aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    #
    # ap1 = load_timbre('/home/sean/pythonProj/torch_npss/data/timbre_model/train/ap/nitech_jp_song070_f001_055_ap.npy', 1, ap_max, ap_min)
    #
    # plt.imshow(np.log(np.transpose(ap1)), aspect='auto', origin='bottom', interpolation='none')
    # plt.show()

    # f0 = np.load('./data/prepared_data/f0.npy').astype(np.double)
    # ap = np.load('./data/prepared_data/ap.npy').astype(np.double)
    #
    # ap = pw.decode_aperiodicity(ap, 32000, 1024)
    # # 合成原始语音
    # synthesized = pw.synthesize(f0, sp, ap, 32000, pw.default_frame_period)
    # # 1.输出原始语音
    # sf.write('./data/gen_wav/noise_synthesized.wav', synthesized, 32000)