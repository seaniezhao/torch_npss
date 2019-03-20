import librosa
import pyworld as pw
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
#np.set_printoptions(threshold=np.nan)


wav_path = './data/gen_samples/gen.npy'
code_sp = np.load(wav_path).astype(np.double)

wav_path = '/home/sean/pythonProj/torch_npss/data/timbre_model/train/sp/nitech_jp_song070_f001_014_sp.npy'
code_sp1 = np.load(wav_path).astype(np.double)

spmin = -21.62037003104595
spmax = 5.839000361601009
code_sp = code_sp * (spmax-spmin) + spmin
code_sp1 = code_sp1 * (spmax-spmin) + spmin

sp = pw.decode_spectral_envelope(code_sp, 32000, 1024)
sp1 = pw.decode_spectral_envelope(code_sp1, 32000, 1024)
#print(data)


#print(ap)



plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom',
                       interpolation='none')
plt.show()
plt.imshow(np.log(np.transpose(sp1)), aspect='auto', origin='bottom',
                       interpolation='none')
plt.show()

f0 = np.load('./data/prepared_data/f0.npy').astype(np.double)
ap = np.load('./data/prepared_data/ap.npy').astype(np.double)

ap = pw.decode_aperiodicity(ap, 32000, 1024)
# 合成原始语音
synthesized = pw.synthesize(f0, sp1, ap, 32000, pw.default_frame_period)
# 1.输出原始语音
sf.write('./data/gen_wav/synthesized.wav', synthesized, 32000)