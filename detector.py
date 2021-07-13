import torch
import os
import sys
import cv2
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from models import build_model
from models.utils import fuse_module
from mmcv import Config
from retime import *

cfg_r18_path = "config/pan/pan_r18_vst.py"
cfg_r50_path = "config/pan/pan_r50_vst.py"
cfg_ctw_path = "config/pan/pan_r18_ctw_finetune.py"
cfg_tt_path = "config/pan/pan_r18_tt_finetune.py"

model_r18_path="checkpoints/pan_r18_vst_VinText/checkpoint.pth.tar"
model_r50_path="checkpoints/pan_r50_vst_VinText/checkpoint.pth.tar"
model_tt_path = "checkpoints/pan_r18_tt_finetune.pth.tar"
model_ctw_path = "checkpoints/pan_r18_ctw_finetune.pth.tar"

def scale_aligned_short(img, short_size=640):
	h, w = img.shape[0:2]
	scale = short_size * 1.0 / min(h, w)
	h = int(h * scale + 0.5)
	w = int(w * scale + 0.5)
	if h % 32 != 0:
		h = h + (32 - h % 32)
	if w % 32 != 0:
		w = w + (32 - w % 32)
	img = cv2.resize(img, dsize=(w, h))
	return img

def prepare_test_data(cfg, img):
    img = img[:, :, [2, 1, 0]]
    img_meta = dict(
        org_img_size=torch.tensor([np.array(img.shape[:2])])
    )

    img = scale_aligned_short(img, int(cfg.data.test.short_size))
    img_meta.update(dict(
        img_size=torch.tensor([np.array(img.shape[:2])])
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    data = dict(
        imgs=img.unsqueeze(0),
        img_metas=img_meta
    )

    return data

def load_model(cfg, model_path):
    # model
    model = build_model(cfg.model)
    model = model.cuda()

    if os.path.isfile(model_path):
        print("Loading model '{}'".format(model_path))
        sys.stdout.flush()

        checkpoint = torch.load(model_path)

        d = dict()
        for key, value in checkpoint['state_dict'].items():
            tmp = key[7:]
            d[tmp] = value
        model.load_state_dict(d)
    else:
        print("No checkpoint found at '{}'".format(model_path))
        raise

    # fuse conv and bn
    model = fuse_module(model)

    return model
def load_model_new(cfg_path,type):
    cfg = Config.fromfile(cfg_path)
    if type == "18":
        model = load_model(cfg, model_r18_path)
    elif type == "50":
        model = load_model(cfg, model_r50_path)
    elif type == "tt":
        model = load_model(cfg, model_tt_path)
    elif type == "ctw":
        model = load_model(cfg, model_ctw_path)
    model.eval()
    return model
def load_model50(cfg_path):
    cfg = Config.fromfile(cfg_path)
    model = load_model(cfg,model_r50_path)
    model.eval()
    return model

model18 = load_model_new(cfg_r18_path,'18')
model50 = load_model_new(cfg_r50_path,'50')
modeltt = load_model_new(cfg_tt_path,'tt')
modelctw = load_model_new(cfg_ctw_path,'ctw')

def draw_bbox(img,bboxes):
    for box in bboxes:
        p1,p2 = (box[0],box[1]),(box[2],box[3])
        p3,p4 = (box[4],box[5]),(box[6],box[7])
        cv2.line(img,p1,p2,(0,255,0),2)
        cv2.line(img,p2,p3,(0,255,0),2)
        cv2.line(img,p3,p4,(0,255,0),2)
        cv2.line(img,p4,p1,(0,255,0),2)
    return img
def draw_polygon(img,bboxes):
    for bbox in bboxes:
        x_coors = bbox[::2]
        y_coors = bbox[1::2]
        pair_points = np.array([[x_coors[i], y_coors[i]] for i in range(len(x_coors))], dtype=np.int32)
        cv2.drawContours(img, [pair_points], 0, (0,255,0), 2)
    return img
def detect(img,type_model):
    """Detect text instances in the specific image

    Args:
        img: The input image. Type: cv2 numpy
        cfg_path: Path to config file. Defaults to cfg_r18_path.
        model_path: Path to model weight. Defaults to model_r18_path.

    Returns:
        bboxes: List of bbox. The format of each element is x0,y0,...,x7,y7.
    """    
    if type_model == '18':
        cfg_path=cfg_r18_path 
        model = model18
    elif type_model == 'tt':
        cfg_path=cfg_tt_path 
        model = modeltt
    elif type_model == 'ctw':
        cfg_path=cfg_ctw_path 
        model = modelctw
    elif type_model == '50':
        cfg_path=cfg_r50_path 
        model = model50
    bboxes = []

    cfg = Config.fromfile(cfg_path)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=False
        ))

    # prepare input
    data = prepare_test_data(cfg, img)
    data['imgs'] = data['imgs'].cuda()
    data.update(dict(
        cfg=cfg
    ))

    # forward
    start = time.time()
    with torch.no_grad():
        outputs = model(**data)
    end = time.time()
    time_process = time_2(end-start,type_model)

    out_bboxes = outputs['bboxes']
    for i, bbox in enumerate(out_bboxes):
        values = [int(v) for v in bbox]
        bboxes.append(values)
    if type_model == 'tt' or type_model == 'ctw':
        img = draw_polygon(img,bboxes)
    else:
        img = draw_bbox(img,bboxes)
    return img,len(bboxes),time_process

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    img,box,_ = detect(img,'tt')
    # print(len(box))
    cv2.imwrite('test_result.jpg',img)