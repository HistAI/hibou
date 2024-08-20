# -*- coding: utf-8 -*-
# CellViT networks and adaptions, without sharing encoders
#
# UNETR paper and code: https://github.com/tamasino52/UNETR
# SAM paper and code: https://segment-anything.com/
#
# PostProcessing Pipeline adapted from HoverNet
# HoverNet Network (https://doi.org/10.1016/j.media.2019.101563)
# Code Snippet adapted from HoverNet implementation (https://github.com/vqdang/hover_net)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
#
# Portions Copyright (c) HistAI Inc.

from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed

from .. import build_model


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DetectionCellPostProcessor:
    def __init__(
        self,
        nr_types: int = None,
        magnification: Literal[20, 40] = 40,
        gt: bool = False,
    ) -> None:
        """DetectionCellPostProcessor for postprocessing prediction maps and get detected cells

        Args:
            nr_types (int, optional): Number of cell types, including background (background = 0). Defaults to None.
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.
            gt (bool, optional): If this is gt data (used that we do not suppress tiny cells that may be noise in a prediction map).
                Defaults to False.

        Raises:
            NotImplementedError: Unknown magnification
        """
        self.nr_types = nr_types
        self.magnification = magnification
        self.gt = gt

        if magnification == 40:
            self.object_size = 10
            self.k_size = 21
        elif magnification == 20:
            self.object_size = 3  # 3 or 40, we used 5
            self.k_size = 11  # 11 or 41, we used 13
        else:
            raise NotImplementedError("Unknown magnification")
        if gt:  # to not supress something in gt!
            self.object_size = 100
            self.k_size = 21

    def post_process_cell_segmentation(
        self,
        pred_map: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Post processing of one image tile

        Args:
            pred_map (np.ndarray): Combined output of tp, np and hv branches, in the same order. Shape: (H, W, 4)

        Returns:
            Tuple[np.ndarray, dict]:
                np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
                dict: Instance dictionary. Main Key is the nuclei instance number (int), with a dict as value.
                    For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
                    contour, type_prob (probability), type (nuclei type)
        """
        if self.nr_types is not None:
            pred_type = pred_map[..., :1]
            pred_inst = pred_map[..., 1:]
            pred_type = pred_type.astype(np.int32)
        else:
            pred_inst = pred_map

        pred_inst = np.squeeze(pred_inst)
        pred_inst = self.__proc_np_hv(
            pred_inst, object_size=self.object_size, ksize=self.k_size
        )

        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

        #### * Get class of each instance id, stored at index id-1 (inst_id = number of deteced nucleus)
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        return pred_inst, inst_info_dict

    def __proc_np_hv(
        self, pred: np.ndarray, object_size: int = 10, ksize: int = 21
    ) -> np.ndarray:
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21
        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """
        pred = np.array(pred, dtype=np.float32)

        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32)

        blb = measurements.label(blb)[0]  # ndimage.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)  # 10
        blb[blb > 0] = 1  # background is 0 already

        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # ksize = int((20 * scale_factor) + 1) # 21 vs 41
        # obj_size = math.ceil(10 * (scale_factor**2)) #10 vs 40

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * blb
        ## nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred


class CellViT(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with vision transformer as backbone encoder

    Skip connections are shared between branches, but each network has a distinct encoder

    The modell is having multiple branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_binary_map: Binary nuclei prediction
        * hv_map: HV-prediction to separate isolated instances
        * nuclei_type_map: Nuclei instance-prediction
        * [Optional, if regression loss]:
        * regression_map: Regression map for binary prediction

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = 14
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.encoder = None

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        self.regression_loss = regression_loss
        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            2 + offset_branches
        )  # todo: adapt for helper loss
        self.hv_map_decoder = self.create_upsampling_branch(
            2
        )  # todo: adapt for helper loss
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
        """
        # assert (
        #     x.shape[-2] % self.patch_size == 0
        # ), "Img must have a shape of that is divisible by patch_size (token_size)"
        # assert (
        #     x.shape[-1] % self.patch_size == 0
        # ), "Img must have a shape of that is divisible by patch_size (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        # z0 = torchvision.transforms.functional.resize(z0, (256,256))

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        # patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        patch_dim = [16, 16]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )
        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    def freeze_encoder(self):
        """Freeze encoder to not train it"""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":  # do not freeze head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def unfreeze_output_layers(self):
        """Unfreeze output layers to train them"""
        extract_layers = [f"blocks.{i - 1}" for i in self.extract_layers]
        # extract_layers += [f'blocks.{i - 2}' for i in self.extract_layers]
        print("*" * 50)
        print(
            "Trainable parameters before unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        for layer_name, p in self.encoder.named_parameters():
            for el in extract_layers:
                if el in layer_name:
                    p.requires_grad = True
        print(
            "Trainable parameters after unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        print("*" * 50)

    def unfreeze_output_layers_2(self):
        """Unfreeze output layers to train them"""
        extract_layers = [f"blocks.{i - 1}" for i in self.extract_layers]
        extract_layers += [f"blocks.{i - 2}" for i in self.extract_layers]
        print("*" * 50)
        print(
            "Trainable parameters before unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        for layer_name, p in self.encoder.named_parameters():
            for el in extract_layers:
                if el in layer_name:
                    p.requires_grad = True
        print(
            "Trainable parameters after unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        print("*" * 50)

    def unfreeze_output_layers_3(self):
        """Unfreeze output layers to train them"""
        extract_layers = [f"blocks.{i - 1}" for i in self.extract_layers]
        extract_layers += [f"blocks.{i - 2}" for i in self.extract_layers]
        extract_layers += [f"blocks.{i - 3}" for i in self.extract_layers]
        print("*" * 50)
        print(
            "Trainable parameters before unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        for layer_name, p in self.encoder.named_parameters():
            for el in extract_layers:
                if el in layer_name:
                    p.requires_grad = True
        print(
            "Trainable parameters after unfreezing output layers:",
            sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
        )
        print("*" * 50)


class HibouEncoder(nn.Module):
    def __init__(
        self,
        path=None,
        extract_layers=[6, 12, 18, 24],
        num_classes=0,
        dropout_rate=0,
        attention_dropout_rate=0,
    ):
        super().__init__()
        self.path = path
        self.encoder = build_model(
            weights_path=path,
            img_size=224,
            arch="vit_large",
            patch_size=14,
            layerscale=1e-5,
            ffn_layer="swiglufused",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            num_register_tokens=4,
            interpolate_offset=0,
            interpolate_antialias=True,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.extract_layers = extract_layers
        self.head = nn.Linear(1024, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = torchvision.transforms.functional.resize(x, (224, 224))
        output = self.encoder.forward_features(x, return_intermediate=True)
        intermediate = output["intermediate"]
        intermediate = [intermediate[i - 1] for i in self.extract_layers]
        intermediate = [
            torch.cat((intermediate[i][:, :1], intermediate[i][:, 5:]), dim=1)
            for i in range(len(intermediate))
        ]
        return self.head(output["x_norm_clstoken"]), None, intermediate


class CellViTHibou(CellViT):
    """CellViT with Hibou backbone settings (https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth)

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model256_path (Union[Path, str]): Path to ViT 256 backbone model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        hibou_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=embed_dim,
            input_channels=input_channels,
            depth=depth,
            num_heads=num_heads,
            extract_layers=extract_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            regression_loss=regression_loss,
        )

        self.hibou_path = hibou_path
        self.encoder = HibouEncoder(
            path=hibou_path,
            extract_layers=self.extract_layers,
            num_classes=num_tissue_classes,
            dropout_rate=drop_rate,
            attention_dropout_rate=attn_drop_rate,
        )
