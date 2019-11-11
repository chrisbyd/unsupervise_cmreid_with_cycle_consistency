import torch.utils.data as data
from .sysu_dataset import SYSUDataset
import torchvision.transforms as transforms


class TrainDatasetDataLoader():
    """
    wrap class for the data loader
    """
    def __init__(self,config):
        self.config = config
        self.get_transform()
        self.construct_dataset()

        print('dataset [%s] was created '%type(self.dataset).__name__)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle= not config.serial_batches,
            num_workers= int(config.num_threads)

        )

    def get_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((self.config.img_h, self.config.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


    def construct_dataset(self):
        if self.config.dataset_name == 'sysu':
            data_path = './Dataset/SYSU-MM01/'
            self.dataset = SYSUDataset(data_path,transform=self.transform_train)

        elif self.config.dataset_name == 'regdb':
            pass

        else:
            raise NotImplementedError('We didnot implement the dataset for other datasets except regdb and sysu-mm01')

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i* self.config.batch_size >= self.config.max_dataset_size:
                break
            yield data
