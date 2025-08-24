import cv2

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        super().__init__()
        
        self.img_paths = img_paths
        self.labels = labels

        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path = './Data/' + img_path
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)
