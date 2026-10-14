import os

from PIL import Image
from torch.utils.data import Dataset


class FaceDetectionDataset(Dataset):
    IMAGE_DIR = os.environ.get("DATASET_DIR", "datasets/ai_face")

    def __init__(
        self,
        generator_df,
        transform=None,
        target_transform=None,
        get_demographic_attributes=False,
    ):
        self.data = generator_df
        self.transform = transform
        self.target_transform = target_transform
        self.return_demographic_attributes = get_demographic_attributes

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        if self.data is None:
            raise ValueError("Dataset not loaded")

        img_path = self.data.loc[idx, "image_path"]
        image_path = os.path.join(self.IMAGE_DIR, img_path)
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        label = self.data.loc[idx, "Target"]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        demographic_attributes = (
            self.get_demographic_attributes(idx)
            if self.return_demographic_attributes
            else None
        )
        return image, label, img_path, demographic_attributes

    def get_demographic_attributes(self, idx):
        """
        Generates NumPy array (4, length of dataset) with the demographic attributes
        gender, age, skin_tone and intersection (usually for the test set)

        Returns the aforementioned NumPy array
        """
        # extract predicted demographic characteristics

        return {
            "gender": self.data.iloc[idx, 3],  # 4th column = predicted gender
            "age": self.data.iloc[idx, 4],  # 5th column = predicted age
            "skin_tone": self.data.iloc[idx, 7],  # 8th column = predicted skin tone
            "intersection": self.data.iloc[idx, 10],  # 11th column = intersections
        }
