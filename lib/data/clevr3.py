from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CLEVR3(Dataset):
    def __init__(self, root, mode):
        # path = os.path.join(root, mode)
        self.root = root
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        self.img_paths = []
        for file in os.scandir(os.path.join(root, 'images')):
            img_path = file.path
            self.img_paths.append(img_path)
            
        self.img_paths.sort()

        self.masks = []
        scene_path = os.path.join(root, 'scenes.json')
        with open(scene_path,'r') as r:
            scene_data = json.load(r)['scenes']
        for k,scene in enumerate(scene_data):
            assert scene['image_filename'] == os.path.split(img_paths[k])[-1]
            self.masks.append(annotate_masks(scene))
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
        img = transform(img)

        mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(128, interpolation=Image.NEAREST),
        ])
        
        mask = self.masks[index]
        mask = [np.array(mask_transform(x[:, :, None].astype(np.uint8))) for x in mask]
        mask = np.stack(mask, axis=0)
        mask = torch.from_numpy(mask.astype(np.float)).float()
        
        return img, mask
        
    def __len__(self):
        return len(self.img_paths)

import jaclearn.vision.coco.mask_utils as mask_utils

def _is_object_annotation_available(scene):
    if len(scene['objects']) > 0 and 'mask' in scene['objects'][0]:
        return True
    return False

def _get_object_masks(scene):
    """Backward compatibility: in self-generated clevr scenes, the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks, and stored in `objects_detection`."""
    if 'objects_detection' not in scene:
        return scene['objects']
    if _is_object_annotation_available(scene):
        return scene['objects']
    return scene['objects_detection']

def annotate_masks(scene):
    if 'objects' not in scene and 'objects_detection' not in scene:
        return list()

    masks = [mask_utils.decode(i['mask']).astype('float32') for i in _get_object_masks(scene)]
    return masks