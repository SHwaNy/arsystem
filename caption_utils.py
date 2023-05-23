import pathlib
import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import copy
import pickle

from PIL import Image
from torchvision import transforms
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.models.detection.faster_rcnn
import torch.nn.functional as F

from torchvision.ops import nms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.roi_align import roi_align
from torchvision.ops import MultiScaleRoIAlign
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


#
#
# class describer(nn.Module):
#     def __init__(self, max_len, is_training):
#         super(describer, self).__init__()
#
#         self.is_training = is_training
#         self.max_len = max_len
#
#         self.linear_layers = nn.Sequential(
#             nn.Linear(256 * 7 * 7, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU()
#         )
#
#         self.embedding_layer = nn.Embedding(1000, 512)
#         self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1,
#                     batch_first=True)
#
#         self.fc_layer = nn.Linear(512, 1000)
#         self.special_idx = {
#             '<pad>': 0,
#             '<bos>': 1,
#             '<eos>': 2
#         }
#
#     def forward_train(self, features, trg):
#
#         features = features.reshape((features.size(0), -1))
#
#         x = self.linear_layers(features)
#         _, (h0, c0) = self.lstm(x.unsqueeze(1))
#
#         word_emb = self.embedding_layer(trg['caps'])
#
#         rnn_input_pps = pack_padded_sequence \
#             (word_emb, lengths=trg['caps_len'],
#              batch_first=True, enforce_sorted=False)
#
#         rnn_output_pps, _ = self.lstm(rnn_input_pps, (h0, c0))
#         predicts = self.fc_layer(rnn_output_pps)
#
#         return predicts
#
#
#
#     def forward_test(self, features):
#         batch_size = features.shape[0]
#
#         features = features.reshape((features.size(0), -1))
#
#         x = self.linear_layers(features)
#         _, (h, c) = self.lstm(x.unsqueeze(1))
#
#         predicts = torch.ones(batch_size, self.max_len + 1, dtype=torch.long).to(device) * self.special_idx['<pad>']
#         predicts[:, 0] = torch.ones(batch_size, dtype=torch.long).to(device) * self.special_idx['<bos>']
#
#         keep = torch.arange(features.shape[0], )  # keep track of unfinished sequences
#         if len(keep) != batch_size:
#             keep = torch.arange(batch_size, )
#
#         for i in range(self.max_len):
#             word_emb = self.embedding_layer(predicts[keep, i])  # (valid_batch_size, embed_size)
#
#             _, (h, c) = self.lstm(word_emb.unsqueeze(1), (h, c))  # (num_layers, valid_batch_size, hidden_size)
#             rnn_output = h[-1]
#
#             # out_r = out_r.repeat(1, 17, 1)
#             out_feat = rnn_output.unsqueeze(1)
#
#             pred = self.fc_layer(out_feat).squeeze(1)  # (valid_batch_size, vocab_size)
#             predicts[keep, i + 1] = pred.log_softmax(dim=-1).argmax(dim=-1)
#
#             non_stop = predicts[keep, i + 1] != self.special_idx['<eos>']
#             keep = keep[non_stop]  # update unfinished indices
#             if keep.nelement() == 0:  # stop if all finished
#                 break
#             else:
#                 h = h[:, non_stop, :]
#                 c = c[:, non_stop, :]
#
#         return predicts
#
#     def forward(self, features, trg=None):
#
#         if self.is_training:
#             return self.forward_train(features, trg)
#         else:
#             return self.forward_test(features)
#


class describer_v2(nn.Module):
    def __init__(self, is_training):
        super(describer_v2, self).__init__()

        self.is_training = is_training


        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.ReLU()
        )

    def forward_train(self, features, trg):

        features = features.reshape((features.size(0), -1))

        x = self.linear_layers(features)
        predicts = F.softmax(x)

        return predicts


    def forward_test(self, features):
        features = features.reshape((features.size(0), -1))

        x = self.linear_layers(features)
        predicts = F.softmax(x, dim=1)
        pred = torch.argmax(predicts, dim=1)

        return pred

    def forward(self, features, trg=None):

        if self.is_training:
            return self.forward_train(features, trg)
        else:
            return self.forward_test(features)



def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1,
                             min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
    weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT
    )

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def dre(predicted, image_size, score_thresh=0.5, margin=0.0):
    all_list = []
    out_bboxes = []
    keep = torch.where(predicted['scores'] >= score_thresh)

    if len(keep) == 0:
        return predicted

    predicted['boxes'] = predicted['boxes'][keep]
    predicted['labels'] = predicted['labels'][keep]
    predicted['scores'] = predicted['scores'][keep]

    boxes = predicted['boxes']

    cal_margin = torch.ceil(((boxes[:, 2:] - boxes[:, :2]) * margin))
    boxes[:, :2] = boxes[:, :2] - cal_margin
    boxes[:, 2:] = boxes[:, 2:] + cal_margin

    boxes[boxes < 0] = 0
    boxes[:, [0, 2]][boxes[:, [0, 2]] > image_size[0]] = image_size[0]
    boxes[:, [1, 3]][boxes[:, [1, 3]] > image_size[1]] = image_size[1]

    labels = predicted['labels']
    persons_boxes = list(torch.split(boxes[torch.where(labels == 1)[0]], 1))
    no_persons_boxes = list(torch.split(boxes[torch.where(labels != 1)[0]], 1))

    while len(persons_boxes) != 0:
        target_box = persons_boxes.pop(0)

        # 재귀 A
        list_idx = list()
        for i in range(0, len(no_persons_boxes)):
            if not bbox_iou(target_box, no_persons_boxes[i]) == 0:
                list_idx.append(i)
        list_idx.sort(reverse=True)
        # 재귀 A 끝

        tmp_boxes = list()
        # 재귀 B
        for i in list_idx:
            tmp_boxes.append(no_persons_boxes.pop(i))
        # 재귀 B 끝
        tmp_boxes.append(target_box)
        tmp_boxes = tmp_boxes[::-1]

        idx = 1
        while len(tmp_boxes) > idx:
            target_box = tmp_boxes[idx]
            # 재귀 A
            list_idx = list()
            for i in range(0, len(no_persons_boxes)):
                if not bbox_iou(target_box, no_persons_boxes[i]) == 0:
                    list_idx.append(i)
            list_idx.sort(reverse=True)
            # 재귀 A 끝
            # 재귀 B
            for i in list_idx:
                tmp_boxes.append(no_persons_boxes.pop(i))
            # 재귀 B 끝
            idx += 1
        all_list.append(tmp_boxes)

    for bboxes in all_list:
        out_bboxes.append(torch.cat(
            [torch.min(torch.cat(bboxes)[:, :2], dim=0)[0],
             torch.max(torch.cat(bboxes)[:, 2:], dim=0)[0]]))

    return out_bboxes


def load_image(path_image):
    raw_img = Image.open(path_image).convert("RGB")
    img = transforms.ToTensor()(raw_img).unsqueeze(0).to(device)
    image_size = (img.size(3), img.size(2))

    return img, image_size, raw_img

def cal_distance(boxes):
    # 거리 측정
    try:
        point1 = [483, 289]
        point2 = [885, 278]
        point3 = [1268, 603]
        point4 = [68, 619]
        w, h = 600, 1000
        srcQuad = np.array([point1, point2, point3, point4], np.float32)
        dstQuad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                           np.float32)

        pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)

        tf_positions = []
        positions = [[int((p[0] + p[2]) / 2), int(p[3])] for p in boxes]
        for position in positions:
            tf_position = np.dot(pers, (position[0], position[1], 1))
            tf_position = (
                tf_position[0] / tf_position[2], tf_position[1] / tf_position[2])
            if 0 < tf_position[0] < 600 and 0 < tf_position[1] < 1000:
                tf_positions.append(tf_position)

        tf_positions = np.array(tf_positions)
        cam_position = np.array((300, 1000))

        distances = np.sqrt(np.sum((tf_positions - cam_position) ** 2, axis=1))
        return distances

    except:
        distances = np.ones(len(boxes))
        return distances


def plot_image(image, bboxes, labels, cls2idx):
    image = np.array(image)
    cnt = 0
    y_pnt = 20
    for bbox, label in zip(bboxes, labels):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        cv2.rectangle(image, (x1, y1), (x2, y2), 255, 2)

        cv2.putText(
            image, str(cnt), (int(x1), y1-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(
            image, str(cnt) + " | " + label, (5, y_pnt),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cnt += 1
        y_pnt += 20
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
