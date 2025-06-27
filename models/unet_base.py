import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(dim, dim)

    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device) * -(math.log(10000) / (half-1))
        )
        emb = t[:, None].float() * emb[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return (self.lin(emb))  # (B, dim)


class ResNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First convolution in residual block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution in residual block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution for increasing channels of skip connection
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x  # Save the original input
        
        # Apply the first convolution + ReLU
        x = F.relu(self.norm1(self.conv1(x)))
        
        # Apply the second convolution
        x = self.norm2(self.conv2(x))
        
        # Add the original input (residual connection)
        residual = self.skip_conv(residual)
        x = x + residual  # Skip connection
        x = F.relu(x)  # Apply ReLU after adding the residual
        
        return x


class DownBlock(nn.Module):
    """
    DownBlock used for downsampling the input feature map. This block consists of:
    - A convolutional layer for downsampling
    - Residual convolutions (ResNetConvBlock)
    - An optional pooling layer (2x2 average pooling) for downsampling
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, code_emb_dim, stride=2, use_skip_connections=True, use_pooling=False, use_linear_layer=False, num_heads=8, num_layers=1):
        super().__init__()
        
        self.use_skip_connections = use_skip_connections  # Flag to decide whether to use skip
        self.stride = stride
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize use_pooling properly
        self.use_pooling = use_pooling  # Ensure use_pooling is initialized
        
        # Downsampling method (Pooling or Stride Convolutions)
        if self.use_pooling:
            self.pool = nn.AvgPool2d(2)  # 2x2 average pooling for downsampling
        else:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=self.stride)  # Stride-2 convolution
        
        self.norm = nn.BatchNorm2d(in_channels)
        
        # Residual block in DownBlock
        self.resnet_block = ResNetConvBlock(in_channels, in_channels)
        self.resnet_block_pool = ResNetConvBlock(in_channels, out_channels)
        
        # Embedding layers for time and codeword
        # if in_channels == out_channels: 
        # self.time_emb = nn.Linear(time_emb_dim, out_channels)
        # self.code_emb = nn.Linear(code_emb_dim, out_channels)
        # # # With ReLU activation
        # # self.time_emb = F.relu(nn.Linear(time_emb_dim, out_channels))
        # # self.code_emb = F.relu(nn.Linear(code_emb_dim, out_channels))
        # else:
        self.time_emb = nn.Linear(time_emb_dim, in_channels)
        self.code_emb = nn.Linear(code_emb_dim, in_channels)
        # # With ReLU activation
        # self.time_emb = F.relu(nn.Linear(time_emb_dim, out_channels//2))
        # self.code_emb = F.relu(nn.Linear(code_emb_dim, out_channels//2))

        # Attention mechanism layers
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads=self.num_heads, batch_first=True) for _ in range(self.num_layers)])

        # 1x1 convolution for reducing channels after concatenation
        if self.use_skip_connections:
            self.conv_after_concat = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1)

        # Optional linear layer between DownBlock and UpBlock
        self.use_linear_layer = use_linear_layer
        if self.use_linear_layer:
            self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x, time_emb, code_emb, combined_emb):
        """
        Forward pass for the DownBlock:
        - Apply time and codeword embeddings
        - Perform downsampling using either pooling or stride-based convolution
        - Apply residual convolutions
        - Optionally use skip connections (concatenation with 1x1 convolution)
        """
        
        # Apply NN on time embedding and PCA codeword embedding
        time_emb = self.time_emb(time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape (B, C, 1, 1)
        code_emb = self.code_emb(code_emb).unsqueeze(-1).unsqueeze(-1)  # Shape (B, C, 1, 1)
        
        # Reshape embeddings to match the spatial dimensions of input x (B, C, H, W)
        time_emb = time_emb.expand(-1, -1, x.shape[2], x.shape[3])  # Expand to match (B, C, H, W)
        code_emb = code_emb.expand(-1, -1, x.shape[2], x.shape[3])  # Expand to match (B, C, H, W)
        
        # # No application of NN on embedding
        # time_emb = (time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape (B, C, 1, 1)
        # code_emb = (code_emb).unsqueeze(-1).unsqueeze(-1)  # Shape (B, C, 1, 1)
        
        # print(f"Shape of x : {x.shape}")
        
        # print(f"Shape of time_emb: {time_emb.shape}")
        # print(f"Shape of code_emb: {code_emb.shape}")
        
        # Add the embeddings to the input
        x = x + time_emb + code_emb
        
        x1 = self.resnet_block(x)
        # Concatenate the skip connection (if enabled)
        x1 = x1 + x
        
        
        
        if self.use_skip_connections:
            # Apply 2x2 average pooling for downsampling (if pooling is enabled)
            if self.use_pooling:
                h = self.pool(x1)
            else:
                h = F.relu(self.norm(self.conv(x1)))

            # Residual block to refine features
            h = self.resnet_block(h)

            # # Concatenate the skip connection (if enabled)
            # h = torch.cat([h, x], dim=1)  # Concatenate along the channel dimension
            
            
            x_resized = F.interpolate(x1, size=h.shape[2:], mode='bilinear', align_corners=False)

            # Add the resized skip connection to the current output
            h = h + x_resized
            
            # # Apply 1x1 convolution to reduce channel dimensions after concatenation
            # h = F.relu(self.conv_after_concat(h))
        else:
            # No skip connection, just apply residual block
            h = self.resnet_block_pool(x1)
        
        h = self.resnet_block_pool(h)
        # Apply attention mechanism over the features (apply for num_layers times)
        h = h.flatten(2).permute(0, 2, 1)  # (B, C, H*W) -> (B, H*W, C)
        for attn_layer in self.attn_layers:
            h, _ = attn_layer(h, h, h)  # Attention mechanism
        # h = h.permute(0, 2, 1).reshape(*x.shape)  # Reshape back
        
        # After attention, reshape based on downsampling, not based on the input shape
        # Reshape to the required output shape (B, C, sqrt(H), sqrt(W))
        spatial_dim = int(h.shape[1] ** 0.5)  # Calculate the square root of spatial dimension
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], spatial_dim, spatial_dim)  # Reshape to (B, C, sqrt(H), sqrt(W))


        # Optional linear layer between down and up block
        if self.use_linear_layer:
            h = self.linear(h)
        
        return h


class UpBlock(nn.Module):
    """
    UpBlock used for upsampling the input feature map. This block consists of:
    - A transpose convolutional layer for upsampling
    - Residual convolutions
    - Skip connections (concatenation or addition)
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, code_emb_dim, stride=2, use_skip_connections=True, num_heads=8, num_layers=1):
        super().__init__()
        
        self.use_skip_connections = use_skip_connections
        self.stride = stride
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Upsampling using transpose convolution
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=self.stride, padding=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.upconv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm_ch = nn.BatchNorm2d(out_channels)
        
        # Residual block in DownBlock
        self.resnet_block = ResNetConvBlock(in_channels, in_channels)
        
        # # Embedding layers for time and codeword
        # self.time_emb = (nn.Linear(time_emb_dim, out_channels))
        # self.code_emb = (nn.Linear(code_emb_dim, out_channels))

        # Attention mechanism layers
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(out_channels, num_heads=self.num_heads, batch_first=True) for _ in range(self.num_layers)])

        # 1x1 convolution for reducing channels after concatenation
        if self.use_skip_connections:
            self.conv_after_concat = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1)

    # def forward(self, x, skip, time_emb, code_emb):
    def forward(self, x, skip):
        """
        Forward pass for the UpBlock:
        - Apply time and codeword embeddings
        - Perform upsampling using transpose convolution
        - Optionally use skip connections (concatenate or add)
        - Apply attention mechanism
        """
        # Apply time embedding and PCA codeword embedding
        # time_emb = self.time_emb(time_emb).unsqueeze(-1).unsqueeze(-1)  # Shape (B, C, 1, 1)
        # code_emb = self.code_emb(code_emb).unsqueeze(-1).unsqueeze(-1)  # Shape (B, C, 1, 1)
        
        # Add the embeddings to the input
        # x = x + time_emb + code_emb
        
        x1 = self.resnet_block(x)
        
        h = self.upconv(x1)
        h = F.relu(self.norm(h))
        
        x2 = h
        h = self.upconv_ch(x2)
        h = F.relu(self.norm_ch(h))
        
        if self.use_skip_connections:
            # Concatenate the skip connection (if enabled)
            h = h + skip
            
            # # Apply 1x1 convolution to reduce channel dimensions after concatenation
            # h = F.relu(self.conv_after_concat(h))
        else:
            # If skip is not used, just add the skip connection directly
            h = h 
        
        # # Resize the skip tensor to match the upsampled tensor size
        # skip = F.interpolate(skip, size=h.shape[2:], mode='bilinear', align_corners=False)
        
        
       
       
        # Apply attention mechanism over the features (apply for num_layers times)
        h = h.flatten(2).permute(0, 2, 1)  # (B, C, H*W) -> (B, H*W, C)
        for attn_layer in self.attn_layers:
            h, _ = attn_layer(h, h, h)  # Attention mechanism
        
        # After attention, reshape based on downsampling, not based on the input shape
        # Reshape to the required output shape (B, C, sqrt(H), sqrt(W))
        spatial_dim = int(h.shape[1] ** 0.5)  # Calculate the square root of spatial dimension
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], spatial_dim, spatial_dim)  # Reshape to (B, C, sqrt(H), sqrt(W))
        
        return h
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        # Pointwise Convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x  # Shape (B, out_channels, H, W)



class CondUNet2D(nn.Module):
    def __init__(self, code_dim=512, base_ch=64, time_dim=128, stride=2, num_heads=8, num_layers=1, use_linear_layer=False, encoder=None, neural_encoder_mode=False):
        super().__init__()
        self.neural_encoder_mode = neural_encoder_mode
        self.encoder = encoder


        self.time_emb = TimeEmbedding(time_dim)
        self.code_mlp = nn.Sequential(
            nn.Linear(code_dim, time_dim),
            nn.SELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.emb_channel_reduction = DepthwiseSeparableConv(time_dim, base_ch)
        self.emb_channel_reduction1 = DepthwiseSeparableConv(time_dim, base_ch*2)
        self.emb_channel_reduction2 = DepthwiseSeparableConv(time_dim, base_ch*4)

        self.inc = nn.Sequential(
            nn.Conv2d(2, base_ch, 3, padding=1),
            nn.SELU()
        )

        self.down1 = DownBlock(base_ch, base_ch*2, time_dim, time_dim, stride=2, use_skip_connections=True, use_linear_layer=use_linear_layer)
        self.down2 = DownBlock(base_ch*2, base_ch*4, time_dim, time_dim, stride=2, use_skip_connections=True, use_linear_layer=use_linear_layer)
        self.down3 = DownBlock(base_ch*4, base_ch*8, time_dim, time_dim, stride=2, use_skip_connections=True, use_linear_layer=use_linear_layer)

        self.use_linear_layer = True
        if self.use_linear_layer:
            self.linear = nn.Linear(base_ch, base_ch)

        self.up1 = UpBlock(base_ch*8, base_ch*4, time_dim, time_dim, stride=2, num_heads=num_heads, num_layers=num_layers)
        self.up2 = UpBlock(base_ch*4, base_ch*2, time_dim, time_dim, stride=2, num_heads=num_heads, num_layers=num_layers)
        self.up3 = UpBlock(base_ch*2, base_ch, time_dim, time_dim, stride=2, num_heads=num_heads, num_layers=num_layers)

        self.outc = nn.Sequential(
            nn.Conv2d(base_ch, 2, 1),
            nn.SELU()
        )

    def forward(self, x, t, raw_input=None, encoder=None):
        x = x.view(-1, 2, 32, 32)
        if self.neural_encoder_mode:
            if self.encoder is None or raw_input is None:
                raise ValueError("Neural encoder and raw_input must be provided in neural encoder mode.")
            codeword = self.encoder(raw_input)


        if encoder is not None and raw_input is not None:
            code = encoder(raw_input)
        else:
            raise ValueError("Neural encoder and raw_input must be provided in neural encoder mode.")

        temb = self.time_emb(t)
        cemb = self.code_mlp(code)
        emb = temb + cemb
        emb = emb.unsqueeze(-1).unsqueeze(-1)

        processed_emb = self.emb_channel_reduction(emb)
        processed_emb1 = self.emb_channel_reduction1(emb)
        processed_emb2 = self.emb_channel_reduction2(emb)

        h1 = F.relu(self.inc(x))
        h1 = h1 + processed_emb

        h2 = self.down1(h1, temb, cemb, emb)
        h2 = h2 + processed_emb1
        h3 = self.down2(h2, temb, cemb, emb)
        h3 = h3 + processed_emb2
        h4 = self.down3(h3, temb, cemb, emb)
        h4 = self.linear(h4)

        u3 = self.up1(h4, h3)
        u2 = self.up2(u3, h2)
        u1 = self.up3(u2, h1)
        u1 = u1 + h1

        return self.outc(u1)
    

