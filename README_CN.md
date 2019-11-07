# torch_npss
[English](README.md)

### 一、功能简介

这项目是 [Merlijn Blaauw, Jordi Bonada 的《A Neural Parametric Singing Synthesizer》](https://arxiv.org/abs/1704.03809/) 的pytroch部分实现。它可以根据某些条件合成歌声。一句话简述，这是一个基于深度学习的"AI歌手"。

### 二、试听小例子
[试听](https://soundcloud.com/sean-zhao-236492288/29-test)

<audio id="audio" controls="" preload="none">
<source id="mp3" src="data/gen_wav/29test.wav">
</audio>


### 三、依赖安装 
```
pip install -r requirements.txt
```

### 四、训练数据及测试数据准备
将音频文件和标注文件放到data/raw目录下，然后执行

```
python data/preprocess.py
```
注意根据处理获得的数据调整hparams.py中的condition_channel
###### 如果只想用自己的数据测试：
- 1. 将自己的数据放到data/raw目录下
- 2. 将 data/preprocess.py 中的 custom_test 改为True
- 3. 运行 data/preprocess.py
- 4. inference.py 中的文件名改成自己的文件名
- 5. python inference.py

### 五、模型的训练
```
python train_harmonoc.py
python train_aperoidic.py
python train_vuv.py
```

### 六、生成方式 
```
注：需要生成的标签已经放到了data/timbre_model/test，可以自己生成数据放到test中相应文件夹下
pip install -r requirements.txt 
python inference.py
```

### 七、有任何使用上的问题，或者交流合作加微信：seanweichat 
### 最后： 如果喜欢本项目，请给个star谢谢
