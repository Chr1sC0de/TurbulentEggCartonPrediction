import pathlib as pt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from functools import partial
import torch

def mat_to_torch(
    mat_path,
    input_feature_names  = ["h", "th", "z"],
    output_feature_names = ["wss_z"]
):
    data_dict         = scio.loadmat(mat_path)
    torch_tensor_data = {key: torch.from_numpy(data) for key, data in data_dict.items       () if not key.startswith("_")}
    x                 = torch.stack([torch_tensor_data[key] for key in input_feature_names])
    y                 = torch.stack([torch_tensor_data[key] for key in output_feature_names])

    if len(x.shape) == 2: x = x[None,:]
    if len(y.shape) == 2: y = y[None,:]
    return x, y

class WSSDataset(Dataset):

    def __init__(self, files, transform=None):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load the torch data
        data = torch.load(self.files[idx])
        # apply the transforms
        if self.transform:
                data = self.transform(data)
        return data

class RandomRotate:
    def __init__(self, input_height, x_name="x", y_name="y"):
        self.input_height = input_height
        self.x_name = x_name
        self.y_name = y_name

    def __call__(self, sample):
        x, y = sample[self.x_name], sample[self.y_name]
        roll_val = int(torch.LongTensor(1).random_(0, self.input_height)[0])
        return {
            self.x_name: torch.roll(x, roll_val, 1),
            self.y_name: torch.roll(y, roll_val, 1)
        }

class ConvNormAct(torch.nn.Module):

    def   __init__(self, in_channels, out_channels, kernel_size=3, padding="same", **kwargs):
        super().__init__()
        if padding == "same":
            assert kernel_size//2 == 1
            padding         = kernel_size                                                                         //2
            self.conv       = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
            self.bnorm      = torch.nn.BatchNorm2d(out_channels)
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.activation(self.bnorm(self.conv(x)))

class UpsampleConv(torch.nn.Module):

    def   __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__()
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv     = ConvNormAct(in_channels, out_channels, kernel_size, **kwargs)
    def forward(self,x):
        return self.conv(self.upsample(x))

class UNet(torch.nn.Module):
    def  __init__(self, in_channels, out_channels, base_channels=64, kernel_size=3):
        super().__init__()
        ConvWrapped = partial(ConvNormAct, kernel_size=3)
        # encoding layers
        self.conv1a = ConvWrapped(in_channels, base_channels)
        self.conv1b = ConvWrapped(base_channels, base_channels)
        self.pool_1 = torch.nn.MaxPool2d(2)
        self.conv2a = ConvWrapped(base_channels, 2*base_channels)
        self.conv2b = ConvWrapped(2*base_channels, 2*base_channels)
        self.pool_2 = torch.nn.MaxPool2d(2)
        self.conv3a = ConvWrapped(2*base_channels, 4*base_channels)
        self.conv3b = ConvWrapped(4*base_channels, 4*base_channels)
        # deconding layers
        self.upsample_1  = UpsampleConv(4*base_channels, 2*base_channels)
        self.conv4a      = ConvWrapped(4*base_channels, 2*base_channels)
        self.conv4b      = ConvWrapped(2*base_channels, 2*base_channels)
        self.upsample_2  = UpsampleConv(2*base_channels, base_channels)
        self.conv5a      = ConvWrapped(2*base_channels, base_channels)
        self.conv5b      = ConvWrapped(base_channels, base_channels)
        self.output_conv = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x  = self.conv1a(x)
        x  = self.conv1b(x)
        c1 = x
        x  = self.pool_1(x)
        x  = self.conv2a(x)
        x  = self.conv2b(x)
        c2 = x
        x  = self.pool_2(x)
        x  = self.conv3a(x)
        x  = self.conv3b(x)
        x  = self.upsample_1(x)
        x  = torch.cat([x, c2], dim=1)
        x  = self.conv4a(x)
        x  = self.conv4b(x)
        x  = self.upsample_2(x)
        x  = torch.cat([x, c1], dim=1)
        x  = self.conv5a(x)
        x  = self.conv5b(x)
        return self.output_conv(x)