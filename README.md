# Visual_attention_tf
[![GitHub license](https://img.shields.io/github/license/vinayak19th/Visual_attention_tf?style=for-the-badge)](https://github.com/vinayak19th/Visual_attention_tf/blob/main/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/visual-attention-tf?style=for-the-badge)
![PyPI](https://img.shields.io/pypi/v/visual-attention-tf?color=%238c49e4&style=for-the-badge)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/visual-attention-tf?style=for-the-badge)


A set of image attention layers implemented as custom keras layers that can be imported dirctly into keras


## Currently Implemented layers:
* Pixel Attention : [Efficient Image Super-Resolution Using Pixel Attention(Hengyuan Zhao et al)](https://arxiv.org/abs/2010.01073)
* Channel Attention : [CBAM: Convolutional Block Attention Module(Sanghyun Woo et al)](https://arxiv.org/abs/1807.06521)

## Installation
You can see the projects official pypi page : https://pypi.org/project/visual-attention-tf/
```bash
pip install visual-attention-tf
```
> Use --no-dependencies if you have tensorflow-gpu installed already
# Usage:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from visual_attention import PixelAttention2D , ChannelAttention2D

inp = Input(shape=(1920,1080,3))
cnn_layer = Conv2D(32,3,,activation='relu', padding='same')(inp)

# Using the .shape[-1] to simplify network modifications. Can directly input number of channels as well
Pixel_attention_cnn = PixelAttention2D(cnn_layer.shape[-1])(cnn_layer)
Channel_attention_cnn = ChannelAttention2D(cnn_layer.shape[-1])(cnn_layer)
```