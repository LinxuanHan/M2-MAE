import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
# from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import convolve


def torch2D_Hausdorff_distance(x, y):  # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance

    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)

    return value.max(1)[0].mean()


class HausdorffLoss(nn.Module):
    def __init__(self, p=2):
        super(HausdorffLoss, self).__init__()
        self.p = p

    def torch2D_Hausdorff_distance(self, x, y, p=2):  # Input be like (Batch,1, width,height) or (Batch, width,height)
        x = x.float()
        y = y.float()
        distance_matrix = torch.cdist(x, y, p=p)  # p=2 means Euclidean Distance

        value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
        value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

        value = torch.cat((value1, value2), dim=1)

        return value.max(1)[0].mean()

    def forward(self, x, y):  # Input be like (Batch,height,width)
        loss = self.torch2D_Hausdorff_distance(x, y, self.p)
        return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
                pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
            self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
                pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.detach().cpu().numpy(), target.detach().cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss


if __name__ == "__main__":
    u = torch.Tensor([[[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]]])
    # print(u.shape)
    v = torch.Tensor([[[0.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -3.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]]])

    print("Input shape is (B,W,H):", u.shape, v.shape)
    u = torch.ones([2, 2, 16, 96, 96])
    v = torch.ones([2, 2, 16, 96, 96])
    HD = HausdorffLoss()
    HD1 = HausdorffDTLoss()
    HD2 = HausdorffERLoss()
    hdLoss = HD(u, v)
    hdLoss1 = HD1(u.reshape(u.shape[0], 1, *u.shape[1:]), v.reshape(v.shape[0], 1, *v.shape[1:]))
    hdLoss2 = HD2(u.reshape(u.shape[0], 1, *u.shape[1:]), v.reshape(v.shape[0], 1, *v.shape[1:]))
    # hdLoss = torch2D_Hausdorff_distance(u, v)

    print("Hausdorff Distance is:", hdLoss)
    print("Hausdorff Distance is:", hdLoss1)
    print("Hausdorff Distance is:", hdLoss2)
    # Input shape is (B,W,H): torch.Size([3, 4, 2]) torch.Size([3, 4, 2])
	# Hausdorff Distance is: tensor(2.6667)
	# Hausdorff Distance is: tensor(8.3750)
	# Hausdorff Distance is: tensor(0.6541)
