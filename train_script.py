import time
from model.wavenet_model import *
from data.dataset import NpssDataset
from model.wavenet_training import *
from model_logging import *
from scipy.io import wavfile

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

model = WaveNetModel(output_length=5,
                     dtype=dtype,
                     bias=True)



if use_cuda:
    print("move model to gpu")
    model.cuda()

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = NpssDataset(dataset_file='data/prepared_data/sp.npy',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length)

print('the dataset has ' + str(len(data)) + ' items')



logger = TensorboardLogger(log_interval=200,
                           validation_interval=400,
                           generate_interval=800,
                           log_dir="logs/chaconne_model")

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='./snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=2000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=16,
              epochs=200,
              continue_training_at_step=0)
