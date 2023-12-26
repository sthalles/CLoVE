
import torchvision


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 two_crop=False, return_coord=False):
        super(ImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform)
        self.imgs = self.samples
        self.two_crop = two_crop
        self.return_coord = return_coord

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        images = []
        coords = []

        if self.transform is not None:
            for i in range(2):
                img, coord = self.transform[i](image)
                images.append(img)
                coords.append(coord)

            for _ in range(6):
                img, coord = self.transform[2](image)
                images.append(img)
                coords.append(coord)
        else:
            img = image

        return images, coords, index, target
