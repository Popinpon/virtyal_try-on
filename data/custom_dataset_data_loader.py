import torch.utils.data
from data.base_data_loader import BaseDataLoader
from torch.utils.data.dataset import Subset


# def CreateDataset(opt):
#     dataset = None
#     from data.aligned_dataset import AlignedDataset
#     dataset = AlignedDataset()

#     print("dataset [%s] was created" % (dataset.name()))
#     dataset.initialize(opt)
#     return dataset

def CreateDataset(opt,train_rate=0.9,mode='video'):

    
    dataset=None
    if mode!='video':
        
        if mode=='image':
            from data.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()
        else :
            raise Exception('correct dataset_mode')
    else:
        from data.video_aligned_dataset import VideoAlignedDataset
        dataset = VideoAlignedDataset()



   #print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    n_samples=len(dataset)
    print("dataset [%s] was created" % (dataset.name()),f", data_size {n_samples}")
    train_size=int(n_samples*train_rate)
    subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
    subset2_indices = list(range(train_size,n_samples)) 
    print("val_all num",len(subset2_indices))
    datasets={}
    datasets['train'] = Subset(dataset, subset1_indices)
    datasets['val'] = Subset(dataset, subset2_indices)
    return datasets


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
