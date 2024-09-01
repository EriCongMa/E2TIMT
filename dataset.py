import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

class Batch_Balanced_Dataset_otmi_7(object):
    def __init__(self, opt):
        self.epoch = 0
        log = open(f'{opt.saved_model}/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate_otmi_7(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset_otmi_7(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            print('*' * 50)
            print('after loading data, the total number is {}'.format(total_number_dataset))
            print('_dataset_log: {}'.format(_dataset_log))

            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self._dataset = _dataset
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def __len__(self):
        return len(self._dataset)

    def get_batch(self):
        balanced_batch_images_1 = []
        balanced_batch_images_2 = []
        balanced_batch_images_3 = []
        balanced_batch_src_texts = []
        balanced_batch_tgt_texts = []
        balanced_batch_src_texts_teacher = []
        balanced_batch_tgt_texts_teacher = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image_1, image_2, image_3, src_text, tgt_text, src_text_teacher, tgt_text_teacher = data_loader_iter.next()
                balanced_batch_images_1.append(image_1)
                balanced_batch_images_2.append(image_2)
                balanced_batch_images_3.append(image_3)
                balanced_batch_src_texts += src_text
                balanced_batch_tgt_texts += tgt_text
                balanced_batch_src_texts_teacher += src_text_teacher
                balanced_batch_tgt_texts_teacher += tgt_text_teacher
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image_1, image_2, image_3, src_text, tgt_text, src_text_teacher, tgt_text_teacher = self.dataloader_iter_list[i].next()
                balanced_batch_images_1.append(image_1)
                balanced_batch_images_2.append(image_2)
                balanced_batch_images_3.append(image_3)
                balanced_batch_src_texts += src_text
                balanced_batch_tgt_texts += tgt_text
                balanced_batch_src_texts_teacher += src_text_teacher
                balanced_batch_tgt_texts_teacher += tgt_text_teacher
                self.epoch += 1
            except ValueError:
                pass
                print('Value Error!')

        balanced_batch_images_1 = torch.cat(balanced_batch_images_1, 0)
        balanced_batch_images_2 = torch.cat(balanced_batch_images_2, 0)
        balanced_batch_images_3 = torch.cat(balanced_batch_images_3, 0)

        return balanced_batch_images_1, balanced_batch_images_2, balanced_batch_images_3, balanced_batch_src_texts, balanced_batch_tgt_texts, balanced_batch_src_texts_teacher, balanced_batch_tgt_texts_teacher

def hierarchical_dataset_otmi_7(root, opt, select_data='/'):
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset_otmi_7(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log

class LmdbDataset_otmi_7(Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            print('Check Dataset Numbers in LmdbDataset_otmi: {}'.format(self.nSamples))
            
            self.filtered_index_list = []
            if opt.src_level == "char":
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    try:
                        label_key_1 = 'label-%09d'.encode() % index
                        label_1 = txn.get(label_key_1).decode('utf-8')
                    except:
                        label_key_1 = 'label_1-%09d'.encode() % index
                        label_1 = txn.get(label_key_1).decode('utf-8')

                    label_key_2 = 'label_2-%09d'.encode() % index
                    label_2 = txn.get(label_key_2).decode('utf-8')

                    label_key_3 = 'label_3-%09d'.encode() % index
                    label_3 = txn.get(label_key_3).decode('utf-8')

                    label_key_4 = 'label_4-%09d'.encode() % index
                    label_4 = txn.get(label_key_4).decode('utf-8')

                    if len(label_1) > self.opt.src_batch_max_length:
                        continue
                    
                    if len(label_2) > self.opt.tgt_batch_max_length:
                        continue

                    self.filtered_index_list.append(index)
            else:
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    try:
                        label_key_1 = 'label-%09d'.encode() % index
                        label_1 = txn.get(label_key_1).decode('utf-8')
                        label_1 = label_1.split(' ')
                    except:
                        label_key_1 = 'label_1-%09d'.encode() % index
                        label_1 = txn.get(label_key_1).decode('utf-8')
                        label_1 = label_1.split(' ')

                    label_key_2 = 'label_2-%09d'.encode() % index
                    label_2 = txn.get(label_key_2).decode('utf-8')
                    label_2 = label_2.split(' ')

                    label_key_3 = 'label_3-%09d'.encode() % index
                    label_3 = txn.get(label_key_3).decode('utf-8')
                    label_3 = label_3.split(' ')

                    label_key_4 = 'label_4-%09d'.encode() % index
                    label_4 = txn.get(label_key_4).decode('utf-8')
                    label_4 = label_4.split(' ')
                    
                    if len(label_1) > self.opt.src_batch_max_length:
                        continue
                    
                    if len(label_2) > self.opt.tgt_batch_max_length:
                        continue

                    self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)
            print('After loading LMDB Data, number of smaples: {}'.format(self.nSamples))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            try:
                label_key_1 = 'label-%09d'.encode() % index
                label_1 = txn.get(label_key_1).decode('utf-8')
            except:
                label_key_1 = 'label_1-%09d'.encode() % index
                label_1 = txn.get(label_key_1).decode('utf-8')
            label_key_2 = 'label_2-%09d'.encode() % index
            label_2 = txn.get(label_key_2).decode('utf-8')

            label_key_3 = 'label_3-%09d'.encode() % index
            label_3 = txn.get(label_key_3).decode('utf-8')

            label_key_4 = 'label_4-%09d'.encode() % index
            label_4 = txn.get(label_key_4).decode('utf-8')

            # Image Loading Section 
            img_key_1 = 'image_1-%09d'.encode() % index
            img_key_2 = 'image_2-%09d'.encode() % index
            img_key_3 = 'image_3-%09d'.encode() % index

            img_key_list = [img_key_1, img_key_2, img_key_3]
            img_res_list = []
            
            for img_key in img_key_list:
                imgbuf = txn.get(img_key)

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    if self.opt.rgb:
                        img = Image.open(buf).convert('RGB')  # for color image
                    else:
                        img = Image.open(buf).convert('L')

                except IOError:
                    print(f'Corrupted image for {index}')
                    print('Program is ended. Please check the dataset ...')
                    exit()
                
                img_res_list.append(img)

            if not self.opt.sensitive:
                label_1 = label_1.lower()
                label_2 = label_2.lower()
                label_3 = label_3.lower()
                label_4 = label_4.lower()

            img_1 = img_res_list[0]
            img_2 = img_res_list[1]
            img_3 = img_res_list[2]

        return (img_1, img_2, img_3, label_1, label_2, label_3, label_4)

class RawDataset(Dataset):
    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])

class RawDataset_from_path(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        with open(opt.img_path_file, 'r') as fr:
            for line in fr:
                try:
                    img_path = line.strip().split('\t')[0]
                    self.image_path_list.append(img_path)
                except:
                    img_path = line.strip()
                    self.image_path_list.append(img_path)
        
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class AlignCollate_otmi_7(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images_1, images_2, images_3, src_labels, tgt_labels, src_labels_teacher, tgt_labels_teacher = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images_1[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images_1 = []
            resized_images_2 = []
            resized_images_3 = []

            for image in images_1:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images_1.append(transform(resized_image))

            image_tensors_1 = torch.cat([t.unsqueeze(0) for t in resized_images_1], 0)

            for image in images_2:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images_2.append(transform(resized_image))

            image_tensors_2 = torch.cat([t.unsqueeze(0) for t in resized_images_2], 0)

            for image in images_3:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images_3.append(transform(resized_image))

            image_tensors_3 = torch.cat([t.unsqueeze(0) for t in resized_images_3], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors_1 = [transform(image) for image in images_1]
            image_tensors_1 = torch.cat([t.unsqueeze(0) for t in image_tensors_1], 0)

            image_tensors_2 = [transform(image) for image in images_2]
            image_tensors_2 = torch.cat([t.unsqueeze(0) for t in image_tensors_2], 0)

            image_tensors_3 = [transform(image) for image in images_3]
            image_tensors_3 = torch.cat([t.unsqueeze(0) for t in image_tensors_3], 0)

        return image_tensors_1, image_tensors_2, image_tensors_3, src_labels, tgt_labels, src_labels_teacher, tgt_labels_teacher

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
