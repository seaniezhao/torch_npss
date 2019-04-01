import hparams
from model.wavenet_model import *
from data.dataset import TimbreDataset
from model.timbre_training import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaveNetModel(hparams.create_harmonic_hparams(), device).to(device)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())
data = TimbreDataset(data_folder='data/timbre_model', receptive_field=model.receptive_field, type=0)
print('the dataset has ' + str(len(data)) + ' items')
trainer = ModelTrainer(model=model,
                       dataset=data,
                       lr=0.0005,
                       weight_decay=0.0,
                       snapshot_path='./snapshots/harmonic',
                       snapshot_name='batchsize_128',
                       snapshot_interval=2000,
                       device=device)


print('start training...')
trainer.train(batch_size=128,
              epochs=3000)