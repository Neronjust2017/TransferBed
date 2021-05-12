import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Camelyon17(ImageList):
    """ Camelyon17 Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'H0'``: hospital_0, \
            ``'H1'``: hospital_1, ``'H2'``: hospital_2,  ``'H3'``: hospital_3 and ``'H4'``: hospital_4.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            hospital_0/
                patient_012_node_0/*.jpg
                ...
            hospital_1/
            hospital_2/
            hospital_3/
            hospital_4/
            image_list/
                hospital_0.txt
                hospital_1.txt
                hospital_2.txt
                hospital_3.txt
                hospital_4.txt
    """
    download_list = [
        ("image_list", "image_list.zip", ""),
        ("hospital_0", "hospital_0.tgz", ""),
        ("hospital_1", "hospital_1.tgz", ""),
        ("hospital_2", "hospital_2.tgz", ""),
        ("hospital_3", "hospital_3.tgz", ""),
        ("hospital_4", "hospital_4.tgz", "")
    ]
    image_list = {
        "H0": "image_list/hospital_0.txt",
        "H1": "image_list/hospital_1.txt",
        "H2": "image_list/hospital_2.txt",
        "H3": "image_list/hospital_3.txt",
        "H4": "image_list/hospital_4.txt"
    }
    CLASSES = ['Normal', 'Tumor']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Camelyon17, self).__init__(root, Camelyon17.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())