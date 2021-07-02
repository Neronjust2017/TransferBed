import os
from .segmentation_list import SegmentationList
from PIL import Image
import numpy as np
import torch
from typing import Optional

class BRATS2015(SegmentationList):
    """`BRATS2015 <https://sites.google.com/site/braintumorsegmentation/home/brats2015>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``.
        data_folder (str, optional): Sub-directory of the image. Default: 'RGB'.
        label_folder (str, optional): Sub-directory of the label. Default: 'synthia_mapped_to_cityscapes'.
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~common.vision.transforms.segmentation.Resize`.

     .. note:: In `root`, there will exist following files after downloading.
            ::
                Flair/
                    brats_2013_pat0001_1/*.jpg
                    ...
                T1/
                T1c/
                T2/
                OT/
                image_list/
                    Flair.txt
                    T1.txt
                    T1c.txt
                    T2.txt
                    OT.txt
        """
    # ID_TO_TRAIN_ID = {
    #     3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
    #     15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
    #     8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18
    # }

    CLASSES = ['0', '1', '2', '3', '4']

    image_list = {
        "Flair": "image_list/Flair.txt",
        "T1": "image_list/T1.txt",
        "T1c": "image_list/T1c.txt",
        "T2": "image_list/T2.txt",
    }

    label_list = "image_list/OT.txt"

    def __init__(self, root, task, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        label_list_file = os.path.join(root, self.label_list)

        super(BRATS2015, self).__init__(root, self.CLASSES, data_list_file, label_list_file, data_folder=None,
                                        label_folder=None, id_to_train_id=None, train_id_to_color=None, **kwargs)


    def __getitem__(self, index):
        image_name = self.data_list[index]
        label_name = self.label_list[index]
        image = Image.open(os.path.join(self.root, image_name))
        label = Image.open(os.path.join(self.root, label_name))
        image, label = self.transforms(image, label)

        # remap label
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label = np.asarray(label, np.int64)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.int64)
        if self.id_to_train_id:
            for k, v in self.id_to_train_id.items():
                label_copy[label == k] = v

        return image, label_copy.copy()

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())