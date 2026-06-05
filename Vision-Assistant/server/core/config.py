import torch
import torch.backends.cudnn as cudnn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    cudnn.benchmark = True