from PIL import Image

import torch
import numpy as np
import io
import time
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from dlinputs import paths
from dlinputs import gopen

class ImageNetDataset(object):
    def __init__(self, shard_spec, mini_batch_size=32, num_epochs=300):
        # Fetching shards from Google Cloud
        self.stream = gopen.sharditerator(shard_spec, epochs=num_epochs, decode=True, shuffle=True)
        self.batch_size = mini_batch_size
        self.epochs = num_epochs

        # Transformations to be applied on each sample
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

        self.inputs_cached = None
        self.targets_cached = None

    def getNextSample(self):
        sample = next(self.stream)

        # Each sample is a dict with 'png' and 'cls'
        img = sample['png']
        cls = sample['cls']

        # Class range in the uploaded ImageNet shard is 1 to 1000, changing that to 0 to 1000
        cls = cls-1

        # Convert to a PIL image and apply Transformation (TorchVision transforms requires the input to be PIL image)
        # img = Image.open(io.BytesIO(img))
        img = Image.fromarray(np.uint8(img))
        img = self.transform(img)

        # FIXME: Verify if the below is because of data corruption. If so, shard ImageNet data again.
        if img.size()[0]==1:
            # Some images in sharded ImageNet have incorrect size, skipping them
            # print("SKIPPING image, incorrect size", "key:", sample['__key__'], "size:", str(img.size()), end="\n\n")
            return self.getNextSample()
        elif cls>=1000:
            # If cls size is greater than 1000, then data is corrupted. So, skip
            # print("SKIPPING image, class > 1000", "key:", sample['__key__'], "class:", str(cls), end="\n\n")
            return self.getNextSample()

        return img, cls

    def getNextBatch(self):
        inputs = None
        targets = []

        start_time = time.time()
        
        for i in range(self.batch_size):

            img, cls = self.getNextSample()

            # Append inputs to tensor object
            if inputs is None:
                inputs = torch.Tensor()
                inputs = img
                inputs = inputs.unsqueeze(0)
            else:
                inputs = torch.cat((inputs, img.unsqueeze_(0)), 0)

            targets.append(cls)

        # Convert targets array into a Tensor
        targets = torch.LongTensor(targets)
    
        end_time = time.time()
        time_to_create_batch = end_time - start_time

        self.targets_cached = targets
        self.inputs_cached = inputs

        return inputs, targets, time_to_create_batch