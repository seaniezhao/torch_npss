from model.wavenet_model import *
from data.dataset import NpssDataset
from model.wavenet_training import *
from model_logging import *


def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
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

def generate_audio(model,
                   length=8000):

    return model.generate_fast(length)

def generate_and_log_samples():
    sample_length=2401
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length)

    gen_path = 'data/gen_samples/'
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
    np.save(gen_path+'gen.npy', samples)
    print("audio clips generated")


if __name__ == '__main__':
    # model = load_latest_model_from('snapshots', use_cuda=True)
    # model = torch.load('snapshots/some_model')
    # gen_path = 'data/gen_samples/'
    # if not os.path.exists(gen_path):
    #     os.mkdir(gen_path)
    # np.save(gen_path+'gen.npy', np.zeros(0))
    generate_and_log_samples()

