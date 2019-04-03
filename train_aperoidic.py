import hparams
from model.wavenet_model import *
from data.dataset import TimbreDataset
from model.timbre_training import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaveNetModel(hparams.create_aperiodic_hparams(), device).to(device)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

trainer = ModelTrainer(model=model,
                       data_folder='data/timbre_model',
                       lr=0.0005,
                       weight_decay=0.0,
                       snapshot_path='./snapshots/aperiodic',
                       snapshot_name='aper',
                       snapshot_interval=50000,
                       device=device)

#epoch = trainer.load_checkpoint('/Users/zhaowenxiao/pythonProj/torch_npss/snapshots/aperiodic/chaconne_model_1021_2019-03-30_09-32-23')

print('start training...')
trainer.train(batch_size=128,
              epochs=3000)
