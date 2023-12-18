# 环境准备

### python=3.8

## 安装torch

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## 安装mmcv

```
pip install mmcv==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
```

## 安装环境

```
pip install -r requirements.txt
```



# 下载权重

链接：https://pan.baidu.com/s/1KZbjxv91Xw_lG7ny-iwu8w?pwd=7m4j 
提取码：7m4j 



# 推理

```
python test.py configs/config.py PATH-TO-CHECKPOINT --format-only --eval-options imgfile_prefix=./result
```

PATH-TO-CHECKPOINT是权重的路径

结果保存在./result

待推理图片的路径请看configs/config.py 110行
