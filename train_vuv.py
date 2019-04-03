import hparams
from model.wavenet_model import *
from data.dataset import TimbreDataset
from model.timbre_training import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaveNetModel(hparams.create_vuv_hparams(), device).to(device)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

trainer = ModelTrainer(model=model,
                        data_folder='data/timbre_model',
                         lr=0.0005,
                         weight_decay=0.0,
                         snapshot_path='./snapshots/vuv',
                         snapshot_name='vuv',
                         snapshot_interval=50000,
                         device=device)

print('start training...')
trainer.train(batch_size=128,
              epochs=1650)