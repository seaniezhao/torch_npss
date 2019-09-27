# torch_npss

[中文(chinese)](README_CN.md)

### implementation of A Neural Parametric Singing Synthesizer: https://arxiv.org/abs/1704.03809
* pretrained models are provided in snapshots/
* generated samples are in data/gen_wav/ 

[sample](https://soundcloud.com/sean-zhao-236492288/29-test)

<audio id="audio" controls="" preload="none">
<source id="mp3" src="data/gen_wav/29test.wav">
</audio>

### try it out!
``` 
pip install -r requirements.txt 
python inference.py
```

### train your own model
- put your audio and label in data/raw
- run data/preprocess.py
- run train_harmonoc.py train_aperoidic.py train_vuv.py 

* if you have any questions feel free to leave an issue
