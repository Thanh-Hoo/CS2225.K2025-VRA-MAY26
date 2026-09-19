from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
from utils.simple_tokenizer import SimpleTokenizer
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True



def read_image(img_list):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if type(img_list) == type("This is a str"):
        img_path = img_list
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                RGB = img.crop((0, 0, 256, 128))
                NI = img.crop((256, 0, 512, 128))
                TI = img.crop((512, 0, 768, 128))
                img3 = [RGB, NI, TI]
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    else:
        img3 = []
        for i in img_list:
            img_path = i
            got_img = False
            if not osp.exists(img_path):
                raise IOError("{} does not exist".format(img_path))
            while not got_img:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img3.append(img)
                    got_img = True
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    pass
    return img3


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid, _, _, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, text_length: int = 77,
                 truncate: bool = True
                 , mask_ratio: float = 0.
                 ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, r_text, n_text, t_text = self.dataset[index]
        img3 = read_image(img_path)
        r_tokens = tokenize(r_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        n_tokens = tokenize(n_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        t_tokens = tokenize(t_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.transform is not None:
            img = [self.transform(img) for img in img3]
        if type(img_path) == type("This is a str"):
            return img, pid, camid, trackid, img_path.split('/')[-1], r_tokens, n_tokens, t_tokens
        else:
            return img, pid, camid, trackid, img_path[0].split('/')[-1], r_tokens, n_tokens, t_tokens
