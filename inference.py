from model.wavenet_model import *
from data.dataset import NpssDataset
from model_logging import *
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np


def load_latest_model_from(location, device):

    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)

    print("load model " + newest_file)

    model = torch.load(newest_file).to(device)

    return model

def get_first_input():

    wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/test/sp/nitech_jp_song070_f001_003_sp.npy'

    code_sp = np.load(wav_path).astype(np.double)
    return torch.Tensor(code_sp).transpose(0, 1)


def get_condition():

    c_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/test/condition/nitech_jp_song070_f001_003_condi.npy'
    conditon = np.load(c_path).astype(np.float)
    return torch.Tensor(conditon).transpose(0, 1)


def generate_audio(model):

    first_input = get_first_input()
    condi = get_condition()
    sample = model.generate(condi, None)
    return sample


def generate_and_log_samples():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_model = load_latest_model_from('snapshots/harmonic', device)

    print("start generating...")
    samples = generate_audio(gen_model)

    gen_path = 'data/gen_samples/'
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
    np.save(gen_path+'gen.npy', samples)
    print("audio clips generated")
    return samples


if __name__ == '__main__':

    gen = generate_and_log_samples().transpose(0,1)

    [sp_min, sp_max, ap_min, ap_max] = np.load('/home/sean/pythonProj/torch_npss/data/timbre_model/min_max_record.npy')

    wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/test/sp/nitech_jp_song070_f001_003_sp.npy'
    load_ = np.load(wav_path).astype(np.double)
    y = gen.cpu().numpy()

    x = torch.sum((torch.Tensor(load_) - gen.cpu()) ** 2)
    print(x)

    load_sp = load_ * (sp_max - sp_min) + sp_min
    sp1 = pw.decode_spectral_envelope(load_sp, 32000, 1024)

    code_sp = y.astype(np.double) * (sp_max - sp_min) + sp_min
    sp = pw.decode_spectral_envelope(code_sp, 32000, 1024)


    plt.imshow(np.log(np.transpose(sp1)), aspect='auto', origin='bottom', interpolation='none')
    plt.show()

    plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom', interpolation='none')
    plt.show()

