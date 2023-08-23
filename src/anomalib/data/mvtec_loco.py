"""MVTec LOCO Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the MVTec LOCO dataset.
    If the dataset is not on the file system, the script downloads and
        extracts the dataset and create PyTorch data objects.
License:
    MVTec LOCO dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).
Reference:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.
    - Paul Bergmann, Killian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection
      and Localization;
      in: International Journal of Computer Vision (130) 947-969, 2022,
      DOI: 10.1007/s11263-022-01578-9.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import albumentations as A
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_loco",
    url="https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701"
    "/mvtec_loco_anomaly_detection.tar.xz",
    hash="d40f092ac6f88433f609583c4a05f56f",
)

CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
)

DEFECT_TYPES = (
    "logical_anomalies",
    "structural_anomalies"
)

def make_mvtec_loco_dataset(
    root: str | Path, split: str | Split | None = None, extensions: Sequence[str] | None = None
) -> DataFrame:
    """Create MVTec LOCO samples by parsing the MVTec LOCO data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/defect_type/image_filename.png
        path/to/dataset/ground_truth/defect_type/image_number/image_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect |  filename.png | ground_truth/defect/image_number      | 1           |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

    Args:
        path (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test). Defaults to None.
        split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.
        create_validation_set (bool, optional): Boolean to create a validation set from the test set.
            MVTec AD dataset does not contain a validation set. Those wanting to create a validation set
            could set this flag to ``True``.

    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

        >>> root = Path('./MVTec_LOCO')
        >>> category = 'breakfast_box'
        >>> path = root / category
        >>> path
        PosixPath('MVTec_LOCO/breakfast_box')

        >>> samples = make_mvtec_loco_dataset(path, split='test', split_ratio=0.1, seed=0)
        >>> samples.head()
           path                     split label image_path                                 mask_path                                               label_index
        0  MVTec_LOCO/breakfast_box test good MVTec_LOCO/breakfast_box/train/good/105.png MVTec_LOCO/breakfast_box/ground_truth/good/105 0
        1  MVTec_LOCO/breakfast_box test good MVTec_LOCO/breakfast_box/train/good/017.png MVTec_LOCO/breakfast_box/ground_truth/good/017 0
        2  MVTec_LOCO/breakfast_box test good MVTec_LOCO/breakfast_box/train/good/137.png MVTec_LOCO/breakfast_box/ground_truth/good/137 0
        3  MVTec_LOCO/breakfast_box test good MVTec_LOCO/breakfast_box/train/good/152.png MVTec_LOCO/breakfast_box/ground_truth/good/152 0
        4  MVTec_LOCO/breakfast_box test good MVTec_LOCO/breakfast_box/train/good/109.png MVTec_LOCO/breakfast_box/ground_truth/good/109 0

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = Path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        raise RuntimeError(f"Found 0 images in {root}")

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL), "mask_path"
    ] = mask_samples.image_path.values

    # assert that the right mask files are associated with the right test images
    assert (
        samples.loc[samples.label_index == LabelName.ABNORMAL]
        .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
        .all()
    ), "Mismatch between anomalous images and ground truth masks. Make sure the mask files in 'ground_truth' \
              folder follow the same naming convention as the anomalous images in the dataset (e.g. image: '000.png', \
              mask: '000.png' or '000_mask.png')."

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTecDataset(AnomalibDataset):
    """MVTec dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str,
        category: str,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.split = split

    def _setup(self) -> None:
        self.samples = make_mvtec_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)


class MVTec(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
        normalize (bool): When True, the images will be normalized to the ImageNet statistics.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to None.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = Path(category)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = MVTecDataset(
            task=task, transform=transform_train, split=Split.TRAIN, root=root, category=category
        )
        self.test_data = MVTecDataset(
            task=task, transform=transform_eval, split=Split.TEST, root=root, category=category
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
