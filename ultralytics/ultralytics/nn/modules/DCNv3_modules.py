import torch
from torch import nn
# from ultralytics.nn.modules.ops_dcnv3.modules import DCNv3  # Cpp implementation
from ultralytics.nn.modules.ops_dcnv3.modules import DCNv3 as DCNv3  # python implementation

from .conv import  Conv
from .block import MaxSigmoidAttnBlock,C2f,C3,C2fAttn,Bottleneck
import torch.nn.functional as F
import torch.autograd

class DCNV3_YoLo(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, group=g, dilation=d,use_dcn_v4_op=False)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x


class Bottleneck_DCNV3(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DCNV3_YoLo(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DCNV3(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fDCNAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))



class CAModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CAModule, self).__init__()
        
        # Define the convolutions after pooling
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2_x = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.conv2_y = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()  
        x_avg_pool_h = torch.mean(x, dim=3, keepdim=True)  # Average across width (W)
        x_avg_pool_w = torch.mean(x, dim=2, keepdim=True)  # Average across height (H)
        x_avg_pool_h = x_avg_pool_h.permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W)
        concat = torch.cat([x_avg_pool_h, x_avg_pool_w], dim=3)  # Concatenate along width dimension
        
        # Apply the first convolution
        out = self.conv1(concat)  # Reduce channel dimension by factor of reduction
        out = self.bn(out)  # Apply Batch Normalization
        out = self.relu(out)  # Apply ReLU non-linear activation
        
        # Split into two paths for width and height attention
        x_path = out[:, :, :, :w]  # X Path (for height)
        y_path = out[:, :, :, w:]  # Y Path (for width)
        
        # Apply convolution and sigmoid to each path
        x_attention = self.conv2_x(x_path)  # Apply to height direction
        y_attention = self.conv2_y(y_path).permute(0, 1, 3, 2)  # Apply to width direction and permute
        
        x_attention = self.sigmoid(x_attention)  # Sigmoid activation for height attention
        y_attention = self.sigmoid(y_attention)  # Sigmoid activation for width attention
        
        out = x * x_attention * y_attention
        
        return out


class C2f_CA(nn.Module):
    def __init__(self, in_channels, out_channels, num_cbs_blocks=1,num_split=1 ,reduction=4):
        super(C2f_CA, self).__init__()
        self.initial_cbs = Conv(in_channels, out_channels,3, 1)
        self.split = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=num_split)
        
        # Split into multiple branches with alternating CBS and CA blocks
        self.branches = nn.ModuleList()
        for _ in range(num_cbs_blocks):
            branch = nn.Sequential(
                Conv(out_channels, out_channels,3, 1),
                Conv(out_channels, out_channels,3, 1),
                CAModule(out_channels, reduction=reduction)
            )
            self.branches.append(branch)
        
        self.concat =Conv(out_channels * (num_cbs_blocks+1), out_channels, 1)
        self.final_cbs = Conv(out_channels, out_channels,3, 1)
    
    def forward(self, x):
        x = self.initial_cbs(x)
        #x = self.split(x)
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(x)
        x = torch.cat(branch_outputs, dim=1)
        x = self.concat(x)
        x = self.final_cbs(x)
        return x


class C2f_CA_DCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_cbs_blocks=1, reduction=4):
        super(C2f_CA_DCN, self).__init__()
        self.initial_cbs = DCNV3_YoLo(in_channels, out_channels,3, 1)
        
        # Split into multiple branches with alternating CBS and CA blocks
        self.branches = nn.ModuleList()
        for _ in range(num_cbs_blocks):
            branch = nn.Sequential(
                DCNV3_YoLo(out_channels, out_channels,3, 1),
                DCNV3_YoLo(out_channels, out_channels,3, 1),
                CAModule(out_channels, reduction=reduction)
            )
            self.branches.append(branch)
        
        self.concat = nn.Sequential(
            nn.Conv2d(out_channels * (num_cbs_blocks+1), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.final_cbs = DCNV3_YoLo(out_channels, out_channels,3, 1)
    
    def forward(self, x):
        # Pass through the initial CBS block
        x = self.initial_cbs(x)
        
        # Split the feature map and process through multiple branches
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(x)
        
        # Concatenate the outputs of the branches along the channel dimension
        x = torch.cat(branch_outputs, dim=1)
        
        # Pass the concatenated output through another CBS block
        x = self.concat(x)
        
        # Final CBS block
        x = self.final_cbs(x)
        
        return x


class Bottleneck_CA2(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.CAModule = CAModule(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        x1=x
        x=self.cv1(x)
        x=self.cv2(x)
        x=x +x1
        x=self.CAModule(x)
        return x 

class C2f_CA2(nn.Module):
    def __init__(self, in_channels, out_channels, num_cbs_blocks=1,num_split=1 ,reduction=4,g=1):
        super(C2f_CA2, self).__init__()
        self.initial_cbs = Conv(in_channels, out_channels,3, 1)
        self.split = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=num_split)
        self.branches = nn.ModuleList()
        for _ in range(num_cbs_blocks):
            branch = nn.Sequential(
                Bottleneck_CA2(out_channels, out_channels, g, k=(3, 3), e=1.0)
            )
            self.branches.append(branch)
        
        # Split into multiple branches with alternating CBS and CA blocks
        
        self.concat =Conv(out_channels * (num_cbs_blocks+1), out_channels, 1)
        self.final_cbs = Conv(out_channels, out_channels,3, 1)
    
    def forward(self, x):
        # Pass through the initial CBS block
        x = self.initial_cbs(x)
        #x = self.split(x)
    
        # # Split the feature map and process through multiple branches
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(x)
        
        
        # Concatenate the outputs of the branches along the channel dimension
        x = torch.cat(branch_outputs, dim=1)
        
        # Pass the concatenated output through another CBS block
        x = self.concat(x)
        
        # Final CBS block
        x = self.final_cbs(x)
        
        return x

class Bottleneck_CA1(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.CAModule = CAModule(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        x1=x
        x=self.cv1(x)
        x=self.cv2(x)
        x=self.CAModule(x)
        return x +x1

class C2f_CA1(nn.Module):
    def __init__(self, in_channels, out_channels, num_cbs_blocks=1,num_split=1 ,reduction=4,g=1):
        super(C2f_CA1, self).__init__()
        self.initial_cbs = Conv(in_channels, out_channels,3, 1)
        self.split = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=num_split)
        self.branches = nn.ModuleList()
        for _ in range(num_cbs_blocks):
            branch = nn.Sequential(
                Bottleneck_CA1(out_channels, out_channels, g, k=(3, 3), e=1.0)
            )
            self.branches.append(branch)
        
        # Split into multiple branches with alternating CBS and CA blocks
        
        self.concat =Conv(out_channels * (num_cbs_blocks+1), out_channels, 1)
        self.final_cbs = Conv(out_channels, out_channels,3, 1)
    
    def forward(self, x):
        # Pass through the initial CBS block
        x = self.initial_cbs(x)
        x = self.split(x)
    
        # # Split the feature map and process through multiple branches
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(x)
        
        
        # Concatenate the outputs of the branches along the channel dimension
        x = torch.cat(branch_outputs, dim=1)
        
        # Pass the concatenated output through another CBS block
        x = self.concat(x)
        
        # Final CBS block
        x = self.final_cbs(x)
        
        return x



class C2f_CA1_DCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_cbs_blocks=1,num_split=1 ,reduction=4,g=1):
        super(C2f_CA1_DCN, self).__init__()
        self.initial_cbs = DCNV3_YoLo(in_channels, out_channels,3, 1)
        self.split = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=num_split)
        self.branches = nn.ModuleList()
        for _ in range(num_cbs_blocks):
            branch = nn.Sequential(
                Bottleneck_CA1(out_channels, out_channels, g, k=(3, 3), e=1.0)
            )
            self.branches.append(branch)
        
        # Split into multiple branches with alternating CBS and CA blocks
        
        self.concat =Conv(out_channels * (num_cbs_blocks+1), out_channels, 1)
        self.final_cbs = DCNV3_YoLo(out_channels, out_channels,3, 1)
    
    def forward(self, x):
        # Pass through the initial CBS block
        x = self.initial_cbs(x)
        x = self.split(x)
    
        # # Split the feature map and process through multiple branches
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(x)
        
        
        # Concatenate the outputs of the branches along the channel dimension
        x = torch.cat(branch_outputs, dim=1)
        
        # Pass the concatenated output through another CBS block
        x = self.concat(x)
        
        # Final CBS block
        x = self.final_cbs(x)
        
        return x

########################################################################################

class C2f_CA2_DCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_cbs_blocks=1,num_split=1 ,reduction=4,g=1):
        super(C2f_CA2_DCN, self).__init__()
        self.initial_cbs = DCNV3_YoLo(in_channels, out_channels,3, 1)
        self.split = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=num_split)
        self.branches = nn.ModuleList()
        for _ in range(num_cbs_blocks):
            branch = nn.Sequential(
                Bottleneck_CA2(out_channels, out_channels, g, k=(3, 3), e=1.0)
            )
            self.branches.append(branch)
        
        # Split into multiple branches with alternating CBS and CA blocks
        
        self.concat =Conv(out_channels * (num_cbs_blocks+1), out_channels, 1)
        self.final_cbs = DCNV3_YoLo(out_channels, out_channels,3, 1)
    
    def forward(self, x):
        # Pass through the initial CBS block
        x = self.initial_cbs(x)
        #x = self.split(x)
    
        # # Split the feature map and process through multiple branches
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(x)
        
        
        # Concatenate the outputs of the branches along the channel dimension
        x = torch.cat(branch_outputs, dim=1)
        
        # Pass the concatenated output through another CBS block
        x = self.concat(x)
        
        # Final CBS block
        x = self.final_cbs(x)
        
        return x
class Bottleneck1(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        x1=x
        x=self.cv1(x)
        x=self.cv2(x)
        if self.add:
          x=x +x1
        return x 

class C2fttn1(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck1(self.c, self.c, True, g, k=(3, 3), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))

class C2fttn2(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck1(self.c, self.c, False, g, k=(3, 3), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))

class C2fCA1ttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CA1(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class C2fCA2ttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CA2(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


import torch.nn.functional as F

class SKA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SKA, self).__init__()
        # Split phase: 3x3 and 5x5 convolution kernels
        self.conv3x3 = Conv(in_channels, in_channels, k=3, p=1)
        self.conv5x5 = Conv(in_channels, in_channels, k=5, p=2)

        # Fuse phase: element-wise summation followed by a global pooling and two FC layers
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels * 2)
        
        # Softmax for selection
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Split
        U1 = self.conv3x3(x)
        U2 = self.conv5x5(x)
        
        # Fuse: element-wise summation
        U = U1 + U2
        
        # Squeeze and Excite with global average pooling
        S = torch.mean(U, dim=(2, 3), keepdim=True)  # Global average pooling
        Z = S.view(S.size(0), -1)
        
        # FC layers
        Z = self.fc1(Z)
        Z = F.relu(Z)
        Z = self.fc2(Z)
        
        # Select: Softmax along channel dimension
        attention_vectors = self.softmax(Z.view(Z.size(0), 2, -1).permute(0, 2, 1))
        
        # Element-wise product with attention vectors
        V1 = attention_vectors[:, :, 0].unsqueeze(2).unsqueeze(3) * U1
        V2 = attention_vectors[:, :, 1].unsqueeze(2).unsqueeze(3) * U2
        
        # Output: Element-wise summation of V1 and V2
        V = V1 + V2
        
        return V


class AMFF_1(nn.Module):
    def __init__(self, in_channels):
        super(AMFF_1, self).__init__()
        self.downsample = nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=2, padding=1)
        self.cbs_p2 = Conv(in_channels[0], in_channels[1])
        self.cbs_p3 = Conv(in_channels[1], in_channels[1])
        self.ska = SKA(in_channels[1])
        
    def forward(self, p):
        # Downsample P2 and pass through CBS
        p2=p[0]#64
        p3=p[1]#128
        #AMFF_2 p2 torch.Size([1, 64, 64, 64])
        #AMFF_2 p3 torch.Size([1, 128, 32, 32])
        # Upsample P3 to match the sp
        p2_downsampled = self.downsample(p2)#[1, 64, 32, 32])
        p2_cbs = self.cbs_p2(p2_downsampled)
        
        # Pass P3 through CBS
        p3_cbs = self.cbs_p3(p3)
        # Fuse the outputs (element-wise addition)
        fused = p2_cbs * p3_cbs
        
        # Pass the fused result through SKA
        output = self.ska(fused)

        
        return output#128

class AMFF_2(nn.Module):
    def __init__(self, in_channels):
        super(AMFF_2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.cbs_p2 = Conv(in_channels[0], in_channels[0])
        self.cbs_p3 = Conv(in_channels[1], in_channels[0])
        self.ska = SKA(in_channels[0])
        
    def forward(self, p):
        #AMFF_2 [64, 128]
        p2=p[0]#64
        p3=p[1]#128
        #AMFF_2 p2 torch.Size([1, 64, 64, 64])
        #AMFF_2 p3 torch.Size([1, 128, 32, 32])
        # Upsample P3 to match the spatial size of P2
        p3_upsampled = self.upsample(p3)#[1, 128, 64, 64]
        p3_cbs = self.cbs_p3(p3_upsampled)#[1, 64, 64, 64]
        
        # Pass P2 through CBS
        p2_cbs = self.cbs_p2(p2)

        
        # Fuse the outputs (element-wise addition)
        fused = p2_cbs * p3_cbs
        
        # Pass the fused result through SKA
        output = self.ska(fused)#AMFF_2 p2 torch.Size([1, 64, 64, 64])
        
        return output




##############################################

class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        #x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)




class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio=2, kernel_size=3):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out



class FeatureAdaptationBlock(nn.Module):
    def __init__(self, c1, c2, n=1):
        super(FeatureAdaptationBlock, self).__init__()
        # Adjusting the in_planes to match the input size
        self.odconv1 = ODConv2d(in_planes=c1, out_planes=c1, kernel_size=1, stride=1, padding=0)
        self.odconv2 = ODConv2d(in_planes=c1, out_planes=2 * c1, kernel_size=1, stride=1, padding=0)
        self.cbam = CBAM(n_channels_in=2 * c1)
        self.ff=nn.Conv2d(2 * c1, c2, 1, bias=True)
    
    def forward(self, x):
        y=x
        x = self.odconv1(x)
        x = self.odconv2(x)
        x = self.cbam(x)
        x=self.ff(x)
        
        return y,x



class AttentionFeatureFusion(nn.Module):
    def __init__(self, c1, c2, n=1, pool_size=4):
        """
        Initialize the AFF module.
        Args:
            in_channels: Number of channels in input feature maps Fh and Fd.
            pool_size: The size of the r x r pooling operation.
        """
        super(AttentionFeatureFusion, self).__init__()
        self.pool_size = pool_size
        
        # Convolution layers
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv_fuse = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
    
    def forward(self, a):
        """
        Forward pass of the AFF module.
        Args:
            F_h: Hazy features (tensor of shape [batch_size, channels, height, width])
            F_d: Dehazed features (tensor of shape [batch_size, channels, height, width])
        Returns:
            F_f: Fused features.
        """
        F_h, F_d=a

        # Step 1: Point-wise addition to fuse Fh and Fd
        X = F_h + F_d
        
        # Step 2: Average Pooling on the fused feature
        X_pooled = F.avg_pool2d(X, kernel_size=self.pool_size)
        
        # Step 3: 3x3 convolution on pooled features
        X_conv = self.conv_fuse(X_pooled)
        
        # Step 4: Bilinear interpolation (upsampling)
        X_upsampled = F.interpolate(X_conv, scale_factor=self.pool_size, mode='bilinear', align_corners=True)
        
        # Step 5: Add shortcut connection
        T = X + X_upsampled
        
        # Step 6: Apply sigmoid to get attention map
        T_sig = torch.sigmoid(T)
        
        # Step 7: Apply 3x3 convolutions to Fh and Fd
        Fh_conv = self.conv1(F_h)
        Fd_conv = self.conv2(F_d)
        
        # Step 8: Element-wise multiplication and combination
        F_f = Fd_conv * T_sig + Fh_conv * (1 - T_sig)
        
        return F_f




class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
class C3k2Attn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))
#class C3k2Attn(C2fAttn):
 #   """Faster Implementation of CSP Bottleneck with 2 convolutions."""

  #  def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        #"""Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
   #     super().__init__(c1, c2, n, shortcut, g, e)
    #    self.m = nn.ModuleList(
     #       C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
      #  )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))




