import torch.utils.data as data
from .sysu_dataset import SYSUDataset
import torchvision.transforms as transforms
from  .data_manager import *
from .test_dataset import *
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
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

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

def get_test_transform(config):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.5])

    transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_h, config.img_w)),
            transforms.ToTensor(),
            normalize,
    ])
    return transform_test

def get_test_dataloader(config):
    transform =get_test_transform(config)
    if config.dataset_name == 'sysu':
        data_path = './Dataset/SYSU-MM01'
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=config.sysu_mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=config.sysu_mode, trial=0)
    elif config.dataset_name =='regdb':
        data_path = './Dataset/RegDB'
        query_img, query_label = process_test_regdb(data_path, trial=config.trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=config.trial, modal='thermal')

    query_dataset = TestData(query_img,query_label,transform=transform,img_size=(config.img_w,config.img_h))
    gallery_dataset = TestData(gall_img, gall_label, transform = transform, img_size =(config.img_w,config.img_h))

    gallery_loader = data.DataLoader(gallery_dataset, batch_size=config.batch_size , shuffle=False,num_workers=int(config.num_threads))
    query_loader = data.DataLoader(query_dataset,batch_size=config.batch_size ,shuffle=False, num_workers= int(config.num_threads))
    nquery = len(query_label)
    print('the length of gallery')

    ngall = len(gall_label)
    print(ngall)
    if config.dataset_name == 'sysu':
        return query_loader,gallery_loader,nquery,ngall ,query_label,gall_label,query_cam,gall_cam
    else:
        return query_loader, gallery_loader, nquery, ngall, query_label, gall_label


