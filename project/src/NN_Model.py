import torch
import torch.nn as nn 

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, (num_classes + 5) * num_anchors, kernel_size=1),
        )
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        B, _, H, W = x.shape
        output = self.pred(x)
        output = output.view(B, self.num_anchors, self.num_classes + 5, H, W)
        output = output.permute(0, 1, 3, 4, 2)
        return output

class YOLO_Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=13, num_anchors=9, grid_ratio=32, scale_factor=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.grid_ratio = grid_ratio
        self.num_anchors = num_anchors

        def s(c):
            return max(8, int(c * scale_factor))

        self.backbone = nn.Sequential(
            CNNBlock(in_channels, s(32), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(32), s(32), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(32), s(64), kernel_size=3, stride=2, padding=1),  # Downsample 1/2
            CNNBlock(s(64), s(64), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(64), s(128), kernel_size=3, stride=2, padding=1), # Downsample 1/4
            CNNBlock(s(128), s(128), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(128), s(256), kernel_size=3, stride=2, padding=1), # Downsample 1/8
            CNNBlock(s(256), s(256), kernel_size=3, stride=1, padding=1),
            CNNBlock(s(256), s(256), kernel_size=3, stride=2, padding=1), # Downsample 1/16
            CNNBlock(s(256), s(256), kernel_size=3, stride=1, padding=1),
        )

        if self.grid_ratio == 32:
            self.extra_down = nn.Sequential(
                CNNBlock(s(256), s(384), kernel_size=3, stride=2, padding=1),  # Downsample 1/32
                CNNBlock(s(384), s(384), kernel_size=3, stride=1, padding=1),
            )
            feature_channels = s(384)
        elif self.grid_ratio == 16:
            self.extra_down = nn.Identity()
            feature_channels = s(256)
        else:
            raise ValueError("Unsupported grid_ratio, only 16 or 32 supported")

        self.head = nn.Sequential(
            CNNBlock(feature_channels, s(256), kernel_size=3, padding=1),
            nn.Dropout(0.1),
            CNNBlock(s(256), s(256), kernel_size=3, padding=1),
        )

        self.scale_pred = nn.Sequential(
            CNNBlock(s(256), s(256), kernel_size=3, padding=1),
            ScalePrediction(s(256), num_classes, num_anchors)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.extra_down(features)
        x = self.head(features)
        output = self.scale_pred(x)
        return output

def init_yolo_weights(model):
	"""
	Complete initialization of YOLO model weights.
	- Conv2d : Kaiming normal (He initialization)
	- Bias : 0 except for objectness which is initialized to -4.6 (~0.01 after sigmoid)
	- BatchNorm : gamma=1, beta=0
	"""
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # also works for SiLU
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)
	
	# Specific initialization for objectness (confidence score)
	for m in model.modules():
		if isinstance(m, ScalePrediction):
			last_conv = m.pred[-1]  # last conv layer in the scale prediction
			if isinstance(last_conv, nn.Conv2d):
				nn.init.kaiming_normal_(last_conv.weight, mode='fan_out', nonlinearity='relu')
				# Bias : objectness at -4.6 (~0.01 after sigmoid), classes to 0
				with torch.no_grad():
					bias = last_conv.bias.view(m.num_anchors, 5 + m.num_classes)
					bias[:, 4].fill_(-4.6)  # Objectness logit
					bias[:, 0:4].fill_(0)   # Tx, Ty, Tw, Th
					bias[:, 5:].fill_(0)    # Class logits
					last_conv.bias.copy_(bias.view(-1))
	return model