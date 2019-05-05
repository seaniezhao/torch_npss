import re
import numpy as np
import matplotlib.pyplot as plt

file = open('harmonic0_0005.log', 'r')

text_content = file.read()
print(text_content)

#train = '(?<=average loss:)\s*\d*\.\d*'
pttern = re.compile(r'(?<=average loss: )\-*\s*\d*\.\d*')
train_loss = np.array(re.findall(pttern, text_content)).astype(np.float32)

t_pttern = re.compile(r'(?<=test loss: )\-*\s*\d*\.\d*')
test_loss = np.array(re.findall(t_pttern, text_content)).astype(np.float32)
test_loss[test_loss>1] /=100000


lst_iter = [i for i in range(1650)]

title = 'weight_decay_loss'
plt.plot(train_loss, '-b', label='train')
plt.plot(test_loss, '-r', label='test')

plt.xlabel("n epoch")
plt.title(title)

# save image
plt.savefig(title+".png")  # should before show method

# show
plt.show()