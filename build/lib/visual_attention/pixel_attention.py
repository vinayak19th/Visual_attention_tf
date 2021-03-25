import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D

class PixelAttention2D(tf.keras.layers.Layer):
    """Implements Pixel Attention ( Hengyuan Zhao et al) for convolutional networks in tensorflow
    Inputs need to be Conv2D feature maps.
    The layer implements the following:
    1. Conv2D with k=1 for fully connected features
    2. Sigmoid activation to create attention maps
    3. tf.multiply to create attention activated outputs
    4. Conv2D with k=1 for fully connected features

    Args:
    * nf [int]: number of filters or channels
    * name : Name of layer
    Call Arguments:
    * Feature maps : Conv2D feature maps of the shape `[batch,W,H,C]`.
    Output;
    Attention activated Conv2D features of shape `[batch,W,H,C]`.

    Here is a code example for using `PixelAttention2D` in a CNN:
    ```python
    inp = Input(shape=(1920,1080,3))
    cnn_layer = Conv2D(32,3,,activation='relu', padding='same')(inp)

    # Using the .shape[-1] to simplify network modifications. Can directly input number of channels as well
    attention_cnn = PixelAttention(cnn_layer.shape[-1])(cnn_layer)

    #ADD DNN layers .....
    ```
    """

    def __init__(self, nf, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.conv1 = Conv2D(filters=nf, kernel_size=1)

    @tf.function
    def call(self, x):
        y = self.conv1(x)
        self.sig = tf.keras.activations.sigmoid(y)
        out = tf.math.multiply(x, y)
        out = self.conv1(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.nf})
        return config