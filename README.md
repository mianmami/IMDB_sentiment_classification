# IMDB影评数据集情感分类
## 说明
1. IMDB数据集和glove预训练词向量文件自行下载放在对应目录下.
2. 先通过执行 python utils.py 构建词表和预训练词向量表,
   然后通过 python run.py --model=TextRNN/TextCNN/DPCNN/fastText --pretrained=True/False
   进行训练和评估.
3. fastText使用Facebook Research的fastText, 参数都保持默认值.
## 实验结果
|model|accuracy|
|:-|:-:|
|TextRNN|0.84348|
|TextCNN|0.85364|
|DPCNN|0.85568|
|fastText|0.87884|