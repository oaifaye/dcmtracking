

import os
import cv2
import torch
import dcmtracking.detection.yolo_fastestv2.model.detector as detector
import dcmtracking.detection.yolo_fastestv2.utils.utils as utils


class YOLO(object):

    def __init__(self, data_path='dcmtracking/detection/yolo_fastestv2/model_data/coco.data',
                 weights_path='dcmtracking/detection/yolo_fastestv2/model_data/coco2017-0.241078ap-model.pth'):
        assert os.path.exists(weights_path), "请指定正确的模型路径"
        assert os.path.exists(data_path), "请指定正确的配置文件路径"
        self.cfg = utils.load_datafile(data_path)
        # 模型加载
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('yolo_fastestv2 device:', self.device)
        self.model = detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def detect_image(self, image):
        '''
        执行检测图片
        :param image:
        :return:
        '''
        res_img = cv2.resize(image, (self.cfg["width"], self.cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.device).float() / 255.0
        preds = self.model(img)
        # 特征图后处理
        output = utils.handel_preds(preds, self.cfg, self.device)
        output_boxes = utils.non_max_suppression(output, conf_thres=0.1, iou_thres=0.4)

        h, w, _ = image.shape
        scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]
        pred_boxes = []
        # 绘制预测框
        for box in output_boxes[0]:
            box = box.tolist()
            obj_score = box[4]
            # print('obj_score:', obj_score)
            category_index = int(box[5])
            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
            pred_boxes.append((x1, y1, x2, y2, category_index, obj_score))
        return pred_boxes
