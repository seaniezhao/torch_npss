from model.wavenet_model import *
from data.dataset import NpssDataset
from model.wavenet_training import *
from model_logging import *
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np

def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)

    #newest_file = 'snapshots/chaconne_model_3000_2019-03-16_11-46-07'
    print("load model " + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model

def get_first_input():
    wav_path = './data/prepared_data/sp.npy'

    code_sp = np.load(wav_path).astype(np.double)
    return torch.Tensor(code_sp).transpose(0, 1)

def get_condition():
    c_path = './data/prepared_data/condition.npy'
    conditon = np.load(c_path).astype(np.float)
    return torch.Tensor(conditon).transpose(0, 1)

def generate_audio(model):

    first_input = get_first_input()
    condi = get_condition()
    return model.generate(condi, first_input)

def generate_and_log_samples():
    gen_model = load_latest_model_from('snapshots', use_cuda=True)


    print("start generating...")
    samples = generate_audio(gen_model)

    gen_path = 'data/gen_samples/'
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
    np.save(gen_path+'gen.npy', samples)
    print("audio clips generated")
    return samples


if __name__ == '__main__':
    # model = load_latest_model_from('snapshots', use_cuda=True)
    # model = torch.load('snapshots/some_model')
    # gen_path = 'data/gen_samples/'
    # if not os.path.exists(gen_path):
    #     os.mkdir(gen_path)
    # np.save(gen_path+'gen.npy', np.zeros(0))
    gen = generate_and_log_samples()
    spmin = -21.62037003104595
    spmax = 5.839000361601009
    code_sp = gen.cpu().numpy().astype(np.double) * (spmax - spmin) + spmin
    sp = pw.decode_spectral_envelope(code_sp, 32000, 1024)

    plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom',
                           interpolation='none')
    plt.show()