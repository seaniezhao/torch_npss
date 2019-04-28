import hparams
from model.wavenet_model import *
from model.timbre_training import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaveNetModel(hparams.create_harmonic_hparams(), device).to(device)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())
trainer = ModelTrainer(model=model,
                       data_folder='data/timbre_model',
                       lr=0.0005,
                       weight_decay=0.0001,
                       snapshot_path='./snapshots/harmonic',
                       snapshot_name='harm0_0001',
                       snapshot_interval=2000,
                       device=device)


def exit_handler():
    trainer.save_model()
    print("exit from keyboard")


#atexit.register(exit_handler)

#epoch = trainer.load_checkpoint('/home/sean/pythonProj/torch_npss/snapshots/harmonic/best_harmonic_model_1649_2019-03-31_17-43-00')

print('start training...')
trainer.train(batch_size=6720,
              epochs=1650)
