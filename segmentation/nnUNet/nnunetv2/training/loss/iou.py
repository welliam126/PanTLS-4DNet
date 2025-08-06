import torch
from torch import nn
from typing import Callable

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, smooth: float = 1.):
        """
        """
        super(IoULoss, self).__init()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        intersection = (x * y).sum()
        union = x.sum() + y.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    iou_loss = IoULoss(apply_nonlin=softmax_helper_dim1, smooth=0)
    iou = iou_loss(pred, ref)
    print(iou)
