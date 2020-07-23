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

### 使用anaconda之后就不需要上面的步骤了

## Keras
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
