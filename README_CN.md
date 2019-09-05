# torch_npss
[English](README.md)

### 一、功能简介

这项目是 [Merlijn Blaauw, Jordi Bonada 的《A Neural Parametric Singing Synthesizer》](!http://example.com/) 的pytroch部分实现。它可以根据某些条件合成歌声。一句话简述，这是一个基于深度学习的"AI歌手"。

### 二、试听小例子
[试听](data/gen_wav/29test.wav)

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
### 五、模型的训练
```
python train_harmonoc.py
python train_aperoidic.py
python train_vuv.py
```

### 六、生成方式 
```
pip install -r requirements.txt 
python inference.py
```

### 七、有任何使用上的问题，或者交流合作加微信：seanweichat
