# 基于胶囊网络的图像描述生成算法

## 硬件
ModelArts平台，硬件配置为`1*Ascend 910 CPU24核 内存96GiB`。
## 环境
- MindSpore 1.7
- numpy
- argparse
- pickle
- json
## 数据集
使用的数据集：MS COCO
使用Faster RCNN提取的图片特征进行训练，数据集获取直接执行`data.ipynb`即可。
## 训练
```shell
用法：python train.py

选项：
    --device               代码实现设备，可选项为Ascend或GPU。默认为Ascend
    --device_id            设备ID，默认为0
    --works                线程数，默认为4
    --lr                   学习率，默认为0.0005
    --batch_size           默认为8
    --result_folder        结果文件存储位置，默认为./ACN_RESULT
    --dataset_path         数据集路径，默认为./mscoco
    --epochs               训练论述，默认为50
    --seq_per_img          每张图片对应句子数量，默认为5
    --encode_layer_num     编码器层数，默认为3
    --decode_layer_num     解码器层数，默认为3
```
