# Visual_attention_tf
A set of image attention layers implemented as custom keras layers that can be imported dirctly into keras


## Currently Implemented layers:
* Pixel Attention : [Efficient Image Super-Resolution Using Pixel Attention(Hengyuan Zhao et al)](https://arxiv.org/abs/2010.01073)
* Channel Attention : [CBAM: Convolutional Block Attention Module(Sanghyun Woo et al)](https://arxiv.org/abs/1807.06521)

# Usage:
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, SeparableConv2D, Concatenate, Multiply, Add
from 
```