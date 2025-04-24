import torch
import torch.nn as nn

class FocusEmbedding(nn.Module):
    def __init__(self, embed_dim, H, W):
        super(FocusEmbedding, self).__init__()
        self.H = H
        self.W = W
        self.embed_dim = embed_dim

        self.fc = nn.Linear(2, H * W) 
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)  

    def forward(self, focus_points):
        B = focus_points.size(0)

        out = self.fc(focus_points)  
        out = out.view(B, 1, self.H, self.W)  

        out = self.conv(out)  #

        return out

