# GMAN
## This is a testing PyTorch version implementation of Graph Multi-Attention Network in the following paper: Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, and Jianzhong Qi. 
## ["GMAN: A Graph Multi-Attention Network for Traffic Prediction", AAAI2020](https://arxiv.org/abs/1911.08415)
## Refer to the implementation of [VincLee8188](https://github.com/VincLee8188/GMAN-PyTorch)

## Requirements
* Python
* PyTorch
* Pandas
* Matplotlib
* Numpy
* gensim
* networkx

## issues
在Perms-bay这个数据集上效果不错，测试结果有MAE：1.93（预测60分钟），但在METR-LA数据集上效果不是很好，不清楚是我的代码实现有问题，还是其他原因，如果有大佬能指出原因，不胜感激


## Citation
This version of implementation is only for learning purpose. For research, please refer to  and  cite from the following paper:
```
@inproceedings{ GMAN-AAAI2020,
  author = "Chuanpan Zheng and Xiaoliang Fan and Cheng Wang and Jianzhong Qi"
  title = "GMAN: A Graph Multi-Attention Network for Traffic Prediction",
  booktitle = "AAAI",
  pages = "1234--1241",
  year = "2020"
}
```


```bash
nohup python main.py --traffic_file ./data/PEMS03.h5 --SE_file ./data/PEMS03-SE.txt --model_file ./data/GMAN-PEMS03-01.pkl --cuda 0 > PEMS03-Iter1.log &
nohup python main.py --traffic_file ./data/PEMS03.h5 --SE_file ./data/PEMS03-SE.txt --model_file ./data/GMAN-PEMS03-02.pkl --cuda 1 > PEMS03-Iter2.log &

nohup python main.py --traffic_file ./data/PEMS04.h5 --SE_file ./data/PEMS04-SE.txt --model_file ./data/GMAN-PEMS04-01.pkl --cuda 2 > PEMS04-Iter1.log &
nohup python main.py --traffic_file ./data/PEMS04.h5 --SE_file ./data/PEMS04-SE.txt --model_file ./data/GMAN-PEMS04-02.pkl --cuda 3 > PEMS04-Iter2.log &
```
