import hparams
from model.wavenet_model import *
from data.dataset import TimbreDataset
from model.timbre_training import *
import atexit

import os
from model_logging import *
from scipy.io import wavfile
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = WaveNetModel(hparams.create_harmonic_hparams(), device).to(device)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

trainer = ModelTrainer(model=model,
                       data_folder='data/timbre_model',
                       lr=0.0001,
                       weight_decay=0.0,
                       snapshot_path='./snapshots/harmonic',
                       snapshot_name='chaconne_model',
                       snapshot_interval=2000,
                       device=device,
                       temperature=0.05)


def exit_handler():
    trainer.save_model()
    print("exit from keyboard")


#atexit.register(exit_handler)

#epoch = trainer.load_checkpoint('/home/sean/pythonProj/torch_npss/snapshots/harmonic/chaconne_model_930_2019-03-26_06-18-49')

print('start training...')
trainer.train(batch_size=128,
              epochs=1650)




# model = WaveNetModel(hparams.create_aperiodic_hparams(), device).to(device)
#
# print('model: ', model)
# print('receptive field: ', model.receptive_field)
# print('parameter count: ', model.parameter_count())
#
# data = TimbreDataset(data_folder='data/timbre_model', receptive_field=model.receptive_field, type=1)
#
# print('the dataset has ' + str(len(data)) + ' items')
#
#
#
# trainer = TimbreTrainer(model=model,
#                          dataset=data,
#                          lr=0.0005,
#                          weight_decay=0.0,
#                          snapshot_path='./snapshots/aperiodic',
#                          snapshot_name='chaconne_model',
#                          snapshot_interval=50000,
#                          device=device)
#
# print('start training...')
# trainer.train(batch_size=32,
#               epochs=420)

