import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import lightning as L
from sklearn import metrics
import math
from pytorch_utils import pad_framewise_output
 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(L.LightningModule):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class ConvPreWavBlock(L.LightningModule):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear layers for Q, K, V for all heads
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Output linear layer
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.init_weight()

    def forward(self, x, dropout_p=0.):
        N, seq_len, embed_size = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape Q, K, V to (N, num_heads, seq_len, head_dim)
        Q = Q.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform scaled dot-product attention and concatenate heads
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_size)

        # Final linear transformation
        return self.fc_out(out)
    
    def init_weight(self):
        init_layer(self.query)
        init_layer(self.key)
        init_layer(self.value)
        init_layer(self.fc_out)


class FXClassifier(L.LightningModule):
    def __init__(self, sample_rate, window_size, hop_size, fmin, 
        fmax, learning_rate,classes_num):
        
        super(FXClassifier, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.learning_rate = learning_rate

        self.loss_func = nn.BCEWithLogitsLoss()

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 256)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=128, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=16, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(128)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        # expansion
        self.conv_block7 = ConvBlock(in_channels=2048, out_channels=4096)

        self.multihead_attention = MultiHeadAttention(embed_size=4096, num_heads=(2**math.ceil(math.log2(classes_num))))
        
        self.fc_pre_audioset = nn.Linear(4096, 4096, bias=True)
        self.fc_audioset = nn.Linear(4096, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc_pre_audioset)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        
        input = torch.as_tensor(input).cuda()
        # Wavegram
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(torch.unsqueeze(input, 1))))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 64, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))
        # Log mel spectrogram
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # Concatenate Wavegram and Log mel spectrogram along the channel dimension
        if (x.size(2) != a1.size(2)):
            min_t = min(x.size(2), a1.size(2))
            min_f = min(x.size(3), a1.size(3))
            x = x[:, :, :min_t, :min_f]
            a1 = a1[:, :, :min_t, :min_f]
        x = torch.cat((x, a1), dim=1)
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block7(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        # [batch_size, 4096, 4 , 2]

        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)

        x = self.multihead_attention(x, dropout_p=.5)

        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x2 + x1

        x = F.relu_(self.fc_pre_audioset(x))

        embedding = F.dropout(x, p=0.5, training=self.training)
        output = self.fc_audioset(x)

        output_dict = {'output': output, 'embedding': embedding}

        return output_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=True)
        return optimizer
    
    def training_step(self, batch_dict, batch_idx):
        inputs = batch_dict["waveform"]
        targets = torch.as_tensor(batch_dict["target"], dtype=torch.float32).cuda()
        output = self(inputs)['output']
        loss = self.loss_func(output, targets)
        self.log('train_loss', loss, on_epoch=True, batch_size=output.size(0))
        self.print(f'Training Loss: {loss.item()}',)
        return loss
    
    def validation_step(self, batch_dict, batch_idx):
        inputs = batch_dict["waveform"]
        targets = torch.as_tensor(batch_dict["target"], dtype=torch.float32).cuda()
        output = self(inputs)['output']
        loss = self.loss_func(output, targets)
        self.log('val_loss', loss, on_epoch=True, batch_size=output.size(0))
        self.print(f'Validation Loss: {loss.item()}')
        
        avg_precision = torch.as_tensor(metrics.average_precision_score(batch_dict["target"], torch.sigmoid(output).to(dtype=torch.float32).numpy(force=True), average=None))
        self.print(f'Validation Average Precision: {avg_precision}')
        self.log("val_avg_precision_across_classes", torch.mean(avg_precision), on_epoch=True, batch_size=output.size(0))