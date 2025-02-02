import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

class ResidualUnit0(nn.Module):
    def __init__(self, channels):
        super().__init__()
        bottleneck = channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv3d(channels, bottleneck, kernel_size=1, padding=0),
            nn.BatchNorm3d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.BatchNorm3d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck, channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.branch1(x) + self.branch2(x)
        return self.relu(out)


class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        bottleneck = channels // 4
        self.branch = nn.Sequential(
            nn.Conv3d(channels, bottleneck, kernel_size=1, padding=0),
            nn.BatchNorm3d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.BatchNorm3d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck, channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.branch(x) + x
        return self.relu(out)


class ResidualBlock(nn.Module):
    """
    A block consisting of one ResidualUnit0 followed by (num_units-1) ResidualUnit layers.
    """
    def __init__(self, channels, num_units):
        super().__init__()
        layers = []
        if num_units > 0:
            layers.append(ResidualUnit0(channels))
            for _ in range(num_units - 1):
                layers.append(ResidualUnit(channels))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)



class MyDecoderBlock3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=1)
        self.res_block = ResidualBlock(out_channel, num_units = 2)

    def forward(self, x, skip=None, depth_scaling=2):
        # Upsample with different scaling for depth dimension
        x = F.interpolate(x, scale_factor=(depth_scaling,2,2), mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.res_block(x)
        return x

class MyUnetDecoder3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip, depth_scaling=[2,2,2,2,2,2]):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s, depth_scaling[i])
            decode.append(d)
        last = d
        return last, decode
        

class EffNetb4(nn.Module):
    def __init__(self, pretrained, decoder_dim=[256, 128, 64, 32, 16]):
        super(EffNetb4, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        num_class=6+1

        self.arch = 'tf_efficientnet_b4.ns_jft_in1k'

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[24,32, 56, 160, 448], # added 24
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(self.arch, [768])
        decoder_dim = decoder_dim

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1]+[0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1],num_class, kernel_size=1)

    def forward(self, batch):
        device = self.D.device

        #image = batch['image'].to(device)
        B, C, D, H, W = batch.shape
        #batch = batch[:,0,:]
        image = batch.reshape(B*D, 1, H, W)

        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        #encode = self.encoder(x)[-5:]
        encode = self.encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])
        #[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

        #[print(f'encode_{i}', e.shape) for i, e in enumerate(encode)]
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=[1,2,2,2,2]
        )
        #print(f'last', last.shape)

        logit = self.mask(last)

        output = logit
        return output

    def encode_for_resnet(self, e, x, B, depth_scaling=[2,2,2,2,1]):

        def pool_in_depth(x, depth_scaling):
            bd, c, h, w = x.shape
            x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
            x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
            x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            return x, x1

        encode=[]
        x = e.conv_stem(x)
        x = e.bn1(x)
        x = e.blocks[0](x)    
        x, x1 = pool_in_depth(x, depth_scaling[0])
        encode.append(x1)
        #print(x.shape)
        #x = e.maxpool(x)
    
        x = e.blocks[1](x)
        x, x1 = pool_in_depth(x, depth_scaling[1])
        encode.append(x1)
        #print(x.shape)
    
        x = e.blocks[2](x)
        x, x1 = pool_in_depth(x, depth_scaling[2])
        encode.append(x1)
        #print(x.shape)
    
        x = e.blocks[3](x)
        x = e.blocks[4](x)
        x, x1 = pool_in_depth(x, depth_scaling[3])
        encode.append(x1)
        #print(x.shape)
    
        x = e.blocks[5](x)
        x = e.blocks[6](x)
        x, x1 = pool_in_depth(x, depth_scaling[4])
        encode.append(x1)
        #print(x.shape)
    
        return encode
