import layers as l
import tensorflow.keras as K


class HarDBlock(K.Model):
    @staticmethod
    def get_link(layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        return out_channels, link

    def __init__(self, base_ch, growth_rate, grmul, n_layers, dwconv=True, keepBase=False):
        super(HarDBlock, self).__init__()
        self.links = []
        layers_ = []
        self.keepBase = keepBase
        self.out_channels = 0
        for i in range(1, n_layers + 1):
            outch, link = self.get_link(i, base_ch, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layers_.append(l.CombConv(outch))
            else:
                layers_.append(l.ConvBNRelu(outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers_list = layers_
        self.concatenate = K.layers.Concatenate(axis=-1)

    def get_out_ch(self):
        return self.out_channels

    def call(self, x):
        layers_ = [x]

        for layer in range(len(self.layers_list)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = self.concatenate(tin)
            else:
                x = tin[0]
            out = self.layers_list[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = self.concatenate(out_)
        return out


class HarDNet(K.Model):
    def __init__(self, depth_wise=False, arch=85, pretrained=True, num_classes=10, weight_path=''):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = []

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            l.ConvBNRelu(filters=first_ch[0], kernels=3,
                       strides=2))

        # Second Layer
        self.base.append(l.ConvBNRelu(first_ch[1], kernels=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(
                K.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same', data_format='channels_last'))
        else:
            self.base.append(l.DWConvBNRelu(strides=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(K.layers.Dropout(0.1))

            self.base.append(l.ConvBNRelu(ch_list[i], kernels=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(
                        K.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last'))
                else:
                    self.base.append(l.DWConvBNRelu(strides=2))
        self.base.append(K.layers.GlobalAveragePooling2D('channels_last'))
        self.base.append(K.layers.Flatten())
        self.base.append(K.layers.Dropout(drop_rate))
        self.base.append(K.layers.Dense(1000))
        self.base.append(K.layers.Dense(num_classes))

    def call(self, x):
        y = x
        for layer in self.base:
            y = layer(y)
        return y


if __name__ == "__main__":
    model = HarDNet(depth_wise=True, arch=39)
    model.build(input_shape=(1, 3, 512, 512))
    print(model.summary())