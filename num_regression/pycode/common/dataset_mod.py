import torch.utils.data as data
from PIL import Image
import numpy as np

class Originaldataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][1]

        roll_str = self.data_list[index][6]
        pitch_str = self.data_list[index][7]

        rad_list = [float(roll_str)*3.141592/180.0, float(pitch_str)*3.141592/180.0]

        img_pil = Image.open(img_path)
        img_pil = img_pil.convert("RGB")
        rad_numpy = np.array(rad_list)

        img_trans_rad, rad_trans = self.transform(img_pil, rad_numpy)


        return img_trans_rad, rad_trans