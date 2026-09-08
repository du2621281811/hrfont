# -*- coding:utf-8 -*-
import cv2
import torch
import settings
from collections import OrderedDict
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from detectionapi import imgproc
from detectionapi.craft import CRAFT
from detectionapi import craft_utils


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


torch.set_num_threads(1)
# load net, initialize
net = CRAFT()
refine_net = None

# Loading weights
net.load_state_dict(copyStateDict(torch.load(settings.detection_model,
                                             map_location=lambda storage, loc: storage.cuda(settings.gpu_id))))
net = net.cuda()
cudnn.benchmark = False

# eval
for p in net.parameters():
    p.requires_grad = False
net.eval()


def det_interface(net, image, text_threshold, link_threshold, low_text, poly, refine_net):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, settings.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=settings.mag_ratio)
    ratio_h = ratio_w = 1 / float(target_ratio)
    with torch.no_grad():
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        x = x.cuda()

        # forward pass
        y, feature = net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            y_refiner = refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes, polys, singlecharboxes = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        singlecharboxes_copy = list()
        for singleboxes in singlecharboxes:
            singleboxes = craft_utils.adjustResultCoordinates(singleboxes, ratio_w, ratio_h)
            singlecharboxes_copy.append(singleboxes)
        # for k in range(len(polys)):
        #     if polys[k] is None: polys[k] = boxes[k]
        #
        # # render results (optional)
        # render_img = score_text.copy()
        # render_img = np.hstack((render_img, score_link))
        # ret_score_text = imgproc.cvt2HeatmapImg(render_img)
        ret_score_text = None

    return boxes, polys, ret_score_text, singlecharboxes_copy


def detection(image):
    #image = imgproc.loadImage(image_path)
    if image is None:
        return None, None, None, None
    #为了显存，兼容近似正方形的图片,截图尺寸为1364/768=1.776
    height, width, channel = image.shape
    # if width / float(height) < settings.imgwidth / float(settings.imgheight):
    #     image_padded = cv2.copyMakeBorder(image, top=0, bottom=0, left=0,
    #                                       right=int(height*settings.imgwidth/float(settings.imgheight))-width,
    #                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # elif width / float(height) > settings.imgwidth / float(settings.imgheight):
    #     image_padded = cv2.copyMakeBorder(image, top=0, bottom=int(width*settings.imgheight/float(settings.imgwidth))-height,
    #                                       left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # else:
    #     image_padded = image
    # if image_padded.shape[1] < settings.imgwidth:
    #     image_padded = cv2.copyMakeBorder(image, top=0, bottom=settings.imgheight-image_padded.shape[0],
    #                                       left=0, right=settings.imgwidth-image_padded.shape[1],
    #                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image_padded = image
    bboxes, polys, score_text, singlecharboxes = det_interface(net, image_padded, settings.text_threshold, settings.link_threshold,
                                         settings.low_text, settings.poly, refine_net)
    # 删除灰度区域的boxes
    bboxes, singlecharboxes = deletebox(bboxes, singlecharboxes, height, width)
    if len(bboxes) == 0:
        return None, None, None, None
    return bboxes, polys, score_text, singlecharboxes


import numpy as np
def delebox(bboxes, singleboxes, height, width):
    final_bboxes = []
    for box in bboxes:
        box[:, 0] = np.clip(box[:, 0], 0, width-1)
        box[:, 1] = np.clip(box[:, 1], 0, height-1)
    for box in bboxes:
        if max(box[:, 1]) - min(box[:, 1]) > 0 and max(box[:, 0]) - min(box[:, 0]) > 0:
            final_bboxes.append(box)

    final_bboxes = np.array(final_bboxes)
    final_singleboxes = []
    for box in singleboxes:
        box[:, 0] = np.clip(box[:, 0], 0, width-1)
        box[:, 1] = np.clip(box[:, 1], 0, height-1)
    for box in singleboxes:
        if max(box[:, 1]) - min(box[:, 1]) > 0 and max(box[:, 0]) - min(box[:, 0]) > 0:
            final_singleboxes.append(box)
    final_singleboxes = np.array(final_singleboxes)
    return final_bboxes, final_singleboxes


def deletebox(bboxes, singlecharboxes, height, width):
    final_bboxes = []
    final_singlecharboxes = []
    for box in bboxes:
        box[:, 0] = np.clip(box[:, 0], 0, width - 1)
        box[:, 1] = np.clip(box[:, 1], 0, height - 1)
    for singleboxes in singlecharboxes:
        for singlebox in singleboxes:
            singlebox[:, 0] = np.clip(singlebox[:, 0], 0, width - 1)
            singlebox[:, 1] = np.clip(singlebox[:, 1], 0, height - 1)

    for i in range(len(bboxes)):
        box = bboxes[i]
        singleboxes = singlecharboxes[i]

        box_copy = list()
        if max(box[:, 1]) - min(box[:, 1]) > 0 and max(box[:, 0]) - min(box[:, 0]) > 0:
            box_copy.append(box)

        singleboxes_copy = list()
        for singlebox in singleboxes:
            if max(singlebox[:, 1]) - min(singlebox[:, 1]) > 0 and max(singlebox[:, 0]) - min(singlebox[:, 0]) > 0:
                singleboxes_copy.append(singlebox)
        if len(box_copy) != 0 and len(singleboxes_copy) != 0:
            final_bboxes.append(box_copy[0])
            final_singlecharboxes.append(singleboxes_copy)
    return final_bboxes, final_singlecharboxes
