import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class PACS(ImageList):
    """ PACS Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Ca'``: Cartoon, ``'Ph'``: Photo and ``'Sk'``: Sketch.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            art_painting/
                dog/*.jpg
                ...
            cartoon/
            photo/
            sketch/
            image_list/
                art_painting.txt
                cartoon.txt
                photo.txt
                sketch.txt
    """
    download_list = [
        ("image_list", "image_list.zip", ""),
        ("art_painting", "Art.tgz", ""),
        ("cartoon", "Clipart.tgz", ""),
        ("photo", "Product.tgz", ""),
        ("sketch", "Real_World.tgz", "")
    ]
    image_list = {
        "Ar": "image_list/art_painting.txt",
        "Ca": "image_list/cartoon.txt",
        "Ph": "image_list/photo.txt",
        "Sk": "image_list/sketch.txt",
    }
    CLASSES = ['horse', 'elephant', 'giraffe', 'person', 'dog', 'guitar', 'house']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(PACS, self).__init__(root, PACS.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())