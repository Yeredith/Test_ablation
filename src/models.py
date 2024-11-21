import torch
import torch.nn as nn
import math
##Revisar modelos por que PSNR es muy bajo
# ==========================
# Componentes Compartidos
# ==========================

class TwoCNN(nn.Module):
    def __init__(self, wn, n_feats=64): 
        super(TwoCNN, self).__init__()

        self.body = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3), stride=1, padding=(1,1)))
               
    def forward(self, x):
    
        out = self.body(x)
        out = torch.add(out, x)
        
        return out             

class ThreeCNN(nn.Module):
    def __init__(self, wn, n_feats=64):
        super(ThreeCNN, self).__init__()
        self.act = nn.ReLU(inplace=True)

        body_spatial = []
        for i in range(2):
            body_spatial.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))

        body_spectral = []
        for i in range(2):
            body_spectral.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))            

        self.body_spatial = nn.Sequential(*body_spatial)
        self.body_spectral = nn.Sequential(*body_spectral)

    def forward(self, x): 
        out = x
        for i in range(2):
              
            out = torch.add(self.body_spatial[i](out), self.body_spectral[i](out))
            if i == 0:
                out = self.act(out)
    
        out = torch.add(out, x)        
        return out
                                                                                                                                                                                                            
class SFCSR(nn.Module):
    def __init__(self, args):
        super(SFCSR, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
 
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
                                                        
        ThreeHead = []
        ThreeHead.append(wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        ThreeHead.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))  
        self.ThreeHead = nn.Sequential(*ThreeHead)
        

        TwoHead = []
        TwoHead.append(wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        self.TwoHead = nn.Sequential(*TwoHead)

        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1))))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1))))
            TwoTail.append(nn.PixelShuffle(3))  

        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1))))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)
                        	 
        twoCNN = []
        for _ in range(self.n_module):
            twoCNN.append(TwoCNN(wn, n_feats))
        self.twoCNN = nn.Sequential(*twoCNN)
        
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))
        
        threeCNN = []
        for _ in range(self.n_module):
            threeCNN.append(ThreeCNN(wn, n_feats))
        self.threeCNN = nn.Sequential(*threeCNN)
      
        reduceD = []
        for _ in range(self.n_module):
            reduceD.append(wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1)) )
        self.reduceD = nn.Sequential(*reduceD)
                                  
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1))   
        
        threefusion = []               
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1))))
        threefusion.append(wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))))          
        self.threefusion = nn.Sequential(*threefusion)
        

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1)) 
        
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1))    
        
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         
        
        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y
        
        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  
    
###########################
           #MCNET
###########################

class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
        super(BasicConv3d, self).__init__()
        self.conv = wn(nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding))
        # Cambiado inplace=True a inplace=False
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class S3Dblock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()

        self.conv = nn.Sequential(
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        )            
       
    def forward(self, x): 
    	   	
        return self.conv(x)

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x
    
    
class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv):
        super(Block, self).__init__()

        # Cambiado inplace=True a inplace=False
        self.relu = nn.ReLU(inplace=False)
        
        Block1 = []  
        for i in range(n_conv):
            Block1.append(S3Dblock(wn, n_feats)) 
        self.Block1 = nn.Sequential(*Block1)         

        Block2 = []  
        for i in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats)) 
        self.Block2 = nn.Sequential(*Block2) 
        
        Block3 = []  
        for i in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats)) 
        self.Block3 = nn.Sequential(*Block3) 
        
        self.reduceF = BasicConv3d(wn, n_feats*3, n_feats, kernel_size=1, stride=1)                                                            
        self.Conv = S3Dblock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))   
         
        conv1 = []   
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv1.append(self.relu)
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv1 = nn.Sequential(*conv1)           

        conv2 = []   
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv2.append(self.relu)
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv2 = nn.Sequential(*conv2)  
        
        conv3 = []   
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv3.append(self.relu)
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv3 = nn.Sequential(*conv3)          
                                                          
    def forward(self, x): 
        res = x
        x1 = self.Block1(x) + x 
        x2 = self.Block2(x1) + x1         
        x3 = self.Block3(x2) + x2     

        x1, depth = _to_4d_tensor(x1, depth_stride=1)  
        x1 = self.conv1(x1)       
        x1 = _to_5d_tensor(x1, depth)  
                             
        x2, depth = _to_4d_tensor(x2, depth_stride=1)  
        x2 = self.conv2(x2)       
        x2 = _to_5d_tensor(x2, depth)         
   
        x3, depth = _to_4d_tensor(x3, depth_stride=1)  
        x3 = self.conv3(x3)       
        x3 = _to_5d_tensor(x3, depth)  
                
        x = torch.cat([self.gamma[0]*x1, self.gamma[1]*x2, self.gamma[2]*x3], 1)                 
        x = self.reduceF(x) 
        x = self.relu(x)
        x = x + res        
        
        x = self.Conv(x)                                                                                                               
        return x  

class MCNet(nn.Module):
    def __init__(self, args):
        super(MCNet, self).__init__()

        scale = args.upscale_factor
        n_colors = args.n_colors
        n_feats = args.n_feats          
        n_conv = args.n_conv
        kernel_size = 3

        band_mean = (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
                     0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
                     0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541)  # CAVE

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, 31, 1, 1])

        if args.cuda:  
            self.band_mean = self.band_mean.cuda()                    

        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size//2))        
        self.SSRM1 = Block(wn, n_feats, n_conv)              
        self.SSRM2 = Block(wn, n_feats, n_conv) 
        self.SSRM3 = Block(wn, n_feats, n_conv)           
        self.SSRM4 = Block(wn, n_feats, n_conv)  
        tail = [
            wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale,2+scale), stride=(1,scale,scale), padding=(1,1,1))),
            wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2))
        ]
        self.tail = nn.Sequential(*tail)

    def forward(self, x, *args):
        """Ajustado para ignorar argumentos adicionales."""
        self.band_mean = self.band_mean.to(x.device)
        x = x - self.band_mean  
        x = x.unsqueeze(1)
        T = self.head(x)
        
        x = self.SSRM1(T)
        x = torch.add(x, T)
        
        x = self.SSRM2(x)
        x = torch.add(x, T)
        
        x = self.SSRM3(x)
        x = torch.add(x, T)
        
        x = self.SSRM4(x)
        x = torch.add(x, T)
        
        x = self.tail(x)
        x = x.squeeze(1)
        
        x = x + self.band_mean.to(x.device)
        
        return x

#################################
           #SFCCBAM
#################################

# Bloque de Atención de Canal con regularización L2
class ChannelAttention(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Regularización L2 aplicada en cada convolución del bloque de atención
        self.fc1 = nn.Conv3d(n_feats, n_feats // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(n_feats // ratio, n_feats, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x

# Bloque CBAM simplificado (solo canal)
class CBAM(nn.Module):
    def __init__(self, n_feats, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(n_feats, ratio)

    def forward(self, x):
        x = self.channel_attention(x)
        return x


# Clase SFCCBAM
class SFCCBAM(nn.Module):
    def __init__(self, args):
        super(SFCCBAM, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
    
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
        # Head 
        ThreeHead = [wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                     wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.ThreeHead = nn.Sequential(*ThreeHead)

        TwoHead = [wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))]
        self.TwoHead = nn.Sequential(*TwoHead)

        # Tail 
        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
            TwoTail.append(nn.PixelShuffle(3))  
        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False)))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)

        # Convoluciones y atenciones
        self.twoCNN = nn.Sequential(*[TwoCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1, bias=False))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))

        self.threeCNN = nn.Sequential(*[ThreeCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD = nn.Sequential(*[wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False)) for _ in range(self.n_module)])                              
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1, bias=False))
        
        threefusion = [wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                       wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.threefusion = nn.Sequential(*threefusion)

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False)) 
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False))    
    
    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         

        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y

        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats  
    
#################################
#######Hybrid-SFCSR##############
#################################

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block"""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_weight = self.global_avg_pool(x)
        se_weight = self.fc1(se_weight)
        se_weight = self.relu(se_weight)
        se_weight = self.fc2(se_weight)
        se_weight = self.sigmoid(se_weight)
        return x * se_weight


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        channel_attention = self.fc1(avg_pool) + self.fc1(max_pool)
        channel_attention = self.relu(channel_attention)
        channel_attention = self.fc2(channel_attention)
        channel_attention = self.sigmoid_channel(channel_attention)
        x = x * channel_attention

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        spatial_attention = self.sigmoid_spatial(spatial_attention)
        x = x * spatial_attention

        return x


class CombinedAttention(nn.Module):
    """Combining SE Block and CBAM Block"""
    def __init__(self, channels, reduction=8, kernel_size=7):
        super(CombinedAttention, self).__init__()
        self.se_block = SEBlock(channels, reduction)
        self.cbam_block = CBAMBlock(channels, reduction, kernel_size)

    def forward(self, x):
        x = self.se_block(x)
        x = self.cbam_block(x)
        return x

class HYBRID_SE_CBAM(nn.Module):
    def __init__(self, args):
        super(HYBRID_SE_CBAM, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats
        self.n_module = args.n_module        
                 
        wn = lambda x: torch.nn.utils.weight_norm(x)
    
        self.gamma_X = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_Y = nn.Parameter(torch.ones(self.n_module)) 
        self.gamma_DFF = nn.Parameter(torch.ones(4))
        self.gamma_FCF = nn.Parameter(torch.ones(2))
        
        # Head 
        ThreeHead = [wn(nn.Conv3d(1, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                     wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.ThreeHead = nn.Sequential(*ThreeHead)

        TwoHead = [wn(nn.Conv2d(1, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))]
        self.TwoHead = nn.Sequential(*TwoHead)

        # Tail 
        TwoTail = []
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*4, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
                TwoTail.append(nn.PixelShuffle(2))           
        else:
            TwoTail.append(wn(nn.Conv2d(n_feats, n_feats*9, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)))
            TwoTail.append(nn.PixelShuffle(3))  
        TwoTail.append(wn(nn.Conv2d(n_feats, 1, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False)))                                 	    	
        self.TwoTail = nn.Sequential(*TwoTail)

        # Convoluciones y atenciones
        self.twoCNN = nn.Sequential(*[TwoCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD_Y = wn(nn.Conv2d(n_feats*self.n_module, n_feats, kernel_size=(1,1), stride=1, bias=False))                          
        self.twofusion = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1), bias=False))

        self.threeCNN = nn.Sequential(*[ThreeCNN(wn, n_feats) for _ in range(self.n_module)])
        self.reduceD = nn.Sequential(*[wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False)) for _ in range(self.n_module)])                              
        self.reduceD_X = wn(nn.Conv3d(n_feats*self.n_module, n_feats, kernel_size=(1,1,1), stride=1, bias=False))
        
        threefusion = [wn(nn.Conv3d(n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)),
                       wn(nn.Conv3d(n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=False))]
        self.threefusion = nn.Sequential(*threefusion)

        self.reduceD_DFF = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_DFF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False)) 
        self.reduceD_FCF = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=(1,1), stride=1, bias=False))  
        self.conv_FCF = wn(nn.Conv2d(n_feats, n_feats, kernel_size=(1,1), stride=1, bias=False))    

        # SE Block and CBAM Block Integration
        self.se_block = SEBlock(n_feats)
        self.cbam_block = CBAMBlock(n_feats)

    def forward(self, x, y, localFeats, i):
        x = x.unsqueeze(1)     
        x = self.ThreeHead(x)    
        skip_x = x         

        y = y.unsqueeze(1)
        y = self.TwoHead(y)
        skip_y = y

        channelX = []
        channelY = []        

        for j in range(self.n_module):        
            x = self.threeCNN[j](x)    
            x = torch.add(skip_x, x)          
            channelX.append(self.gamma_X[j]*x)

            y = self.twoCNN[j](y)           
            y = torch.cat([y, x[:,:,0,:,:], x[:,:,1,:,:], x[:,:,2,:,:]],1)
            y = self.reduceD[j](y)      
            y = torch.add(skip_y, y)         
            channelY.append(self.gamma_Y[j]*y) 
                              
        x = torch.cat(channelX, 1)
        x = self.reduceD_X(x)
        x = self.threefusion(x)
      	                
        y = torch.cat(channelY, 1)        
        y = self.reduceD_Y(y) 
        y = self.twofusion(y)        

        # Apply SE Block
        y = self.se_block(y)
        # Apply CBAM Block
        y = self.cbam_block(y)
     
        y = torch.cat([self.gamma_DFF[0]*x[:,:,0,:,:], self.gamma_DFF[1]*x[:,:,1,:,:], self.gamma_DFF[2]*x[:,:,2,:,:], self.gamma_DFF[3]*y], 1)
       
        y = self.reduceD_DFF(y)  
        y = self.conv_DFF(y)
                       
        if i == 0:
            localFeats = y
        else:
            y = torch.cat([self.gamma_FCF[0]*y, self.gamma_FCF[1]*localFeats], 1) 
            y = self.reduceD_FCF(y)                    
            y = self.conv_FCF(y) 
            localFeats = y  
        y = torch.add(y, skip_y)
        y = self.TwoTail(y) 
        y = y.squeeze(1)   
                
        return y, localFeats