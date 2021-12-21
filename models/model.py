
# Props to Aladdin Persson for this _MUCH_ cleaner implementation.

import math
import torch
import torch.nn as nn


def assert_power_of_two(n: int):
	assert 2**int(math.log(n)/math.log(2)) == n


class DoubleConv(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super(DoubleConv, self).__init__()
		self.op = nn.Sequential(
			# No bias 'cause we're using BatchNorm.  It will get cancelled out.
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(inplace=True),

			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(inplace=True),
		)

	def forward(self, x):
		return self.op(x)

class UNet(nn.Module):
	def __init__(self, in_channels:int = 3, out_channels: int = 3, feature_counts=None):
		super(UNet, self).__init__()

		# Prevent modification of mutable default.
		if feature_counts is None:
			feature_counts = [64, 128, 256, 512]

		# Need nn.ModuleList instead of List for batch evals.
		self.downsamples = nn.ModuleList()
		self.bottleneck = DoubleConv(feature_counts[-1], feature_counts[-1]*2)
		self.upsamples = nn.ModuleList()
		self.finalconv = nn.Conv2d(feature_counts[0], out_channels, kernel_size=1)  # 1x1 conv -> Change # feats.

		# Downsample-Reduction step.
		num_channels = in_channels
		for f_count in feature_counts:
			self.downsamples.append(DoubleConv(in_channels=num_channels, out_channels=f_count))
			num_channels = f_count

		# Up-steps.
		for f_count in reversed(feature_counts):
			# For theses one needs to step by 3 in the upsample step:
			# Use 2x Upscale / Depth-convolve as an operation.
			#self.upsamples.append(nn.UpsamplingBilinear2d(scale_factor=2))
			#self.upsamples.append(nn.Conv2d(f_count*2, f_count, kernel_size=1))
			# Use 4x Upscale / Double-convolve as an operation.
			#self.upsamples.append(nn.UpsamplingNearest2d(scale_factor=4))
			#self.upsamples.append(DoubleConv(f_count*2, f_count))

			# For this one needs to step by two in the upsample step.  (See upsample_step_size)
			# Use ConvTranspose as an operation:
			self.upsamples.append(nn.ConvTranspose2d(f_count*2, f_count, kernel_size=2, stride=2))  # Upscale 2x.

			# Final concatenated convolution:
			self.upsamples.append(DoubleConv(f_count*2, f_count))

	def forward(self, x):
		skip_connections = list()  # Don't need ModuleList because this is not retained.
		for dwn in self.downsamples:
			x = dwn(x)
			skip_connections.append(x)
			x = torch.max_pool2d(x, kernel_size=2, stride=2)

		x = self.bottleneck(x)
		skip_connections.reverse()

		upsample_step_size = 2
		for idx in range(0, len(self.upsamples), upsample_step_size):
			x = self.upsamples[idx+0](x)
			#x = self.upsamples[idx+1](x)
			skip_x = skip_connections[idx//upsample_step_size]

			# It's possible that due to integer division the sizes slightly mismatch.
			#if x.shape[2] != skip_x.shape[2] or x.shape[3] != skip_x.shape[3]:
				# This causes issues withunpacking non-iterables:
				#_, _, h, w = sk.shape
				#x = torchvision.transforms.CenterCrop((h, w))(x)
				# This causes issues with PIL/Tensor mismatch.
				#x = VF.resize(x, size=sk.shape[2:])
				#x = torchvision.transforms.functional.resize(x, size=sk.shape[2:])
				# This seems to work:
				#x = torch.nn.functional.interpolate(x, size=skip_x.shape[2:])

			assert len(skip_x.shape) == 4  # So we don't accidentally unpinch another dimension.
			concat_skip = torch.cat((skip_x, x), dim=1)  # Dim 1 is channel-dimension.  [b, c, h, w]
			x = self.upsamples[idx+1](concat_skip)

		return torch.sigmoid(self.finalconv(x))
