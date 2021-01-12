# ComputerVision
学习TensorFlow和PyTorch的过程记录

## 运行环境
需要在虚拟环境中运行。
```
/home/chenkaixu/venv/bin/python3.6
```

### 在settings.json中设置运行环境
```
"python.pythonPath": "/home/chenkaixu/venv/bin/python3.6",
"python.venvPath": "/home/chenkaixu/venv",
"python.linting.pylintPath": "pylint",
"python.formatting.autopep8Path": "autopep8"
```

**使用anaconda之后就不需要上面的步骤了**

# Keras
gpu支持
```
conda install -c anaconda keras-gpu
```

## 查看gpu支持
```
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```


## 查看详细硬件信息
```
from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())
```

## 查看tensorflow的gpu支持
```
import tensorflow as tf
print (tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
```

# GIT绑定GitHub
## 设置user.name/user.email
```
git config --global user.name "你的用户名"
git config --global user.email "你的公司或个人邮箱"
```

## 生成key
```
ssh-keygen -t rsa -C "你的公司或个人邮箱"
```

## 查看id_rsa.pub
```
cat ~/.ssh/id_rsa.pub 
```

## 复制到GitHub上

## 测试连接
```
ssh -T git@bitbucket.org
ssh git@github.com
``` 

# PyTorch
## 安装
使用window+anconda安装
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
使用pip安装
```
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 测试安装
```
from future import print_function import torch 
x = torch.rand(5, 3) print(x)
```

## 测试是否使用cuda
```
import torch torch.cuda.is_available()
```

# Tensorboard
```
tensorboard --logdir dir
```