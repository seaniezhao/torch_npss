import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import os
#np.set_printoptions(threshold=np.nan)


wav_path = './data/gen_samples/gen.npy'
#wav_path = './data/prepared_data/sp.npy'

code_sp = np.load(wav_path).astype(np.double)*128

sp = pw.decode_spectral_envelope(code_sp, 22050, 1024)
#print(data)


#print(ap)



plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom',
                       interpolation='none')
plt.show()
