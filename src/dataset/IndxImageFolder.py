from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from .image_folder_base import DemographicsMixin


class IndxImageFolder(DemographicsMixin, ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        task_id="",
        num_classes=2,
        allow_empty=False,
        return_demographics=False,
        return_mode="path",  # "path" (legacy) | "index" (prompt method specific)
    ):
        super().__init__(root, transform, target_transform, loader, None, allow_empty)
        self.num_classes = num_classes
        self.return_demographics = return_demographics
        self.task_id = task_id
        self.return_mode = return_mode

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_demographics:
            demographics = self.get_demographics(index)
            if self.return_mode == "path":
                return sample, target, path, demographics
            return index, sample, target, demographics
        if self.return_mode == "index":
            return index, sample, target
        return sample, target, path
