import sys
from dcmtracking.utils.parser import get_config
from dcmtracking.deep_sort import DeepSort
import torch
import cv2
import time
import os
import numpy as np
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
DEEP_SORT_PATH = FILE_PATH.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file(os.path.join(DEEP_SORT_PATH, "deep_sort.yaml"))
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

point_fps = 48 # 画路径的时候，只取后48帧
save_dict = {'cur_num': 0}
ACCIENT_SURE = 2
ACCIENT_WARMING = 1
ACCIENT_NORMAL = 0

def plot_bboxes(image, bboxes, line_thickness=None):

    # Plots one bounding box on image img
    # tl = 3  # line/font thickness
    # for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
    cls_id = ''
    for car in bboxes:
        x1, y1, x2, y2 = car['location']
        lastdistence = car['lastdistence']
        points = car['points']
        track_id = car['track_id']
        speed = car['speed']
        stop = car['stop']
        accident = car['accident']
        angle = car['angle']
        speed_a = car['speed_a']
        angle_a = car['angle_a']
        if accident == ACCIENT_SURE:
            color = (0, 0, 255)
        elif accident == ACCIENT_WARMING:
            # color = (0, 215, 255)
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)

        # 每个车存一个文件夹
        # save_path = 'tmp/'+str(track_id)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(os.path.join(save_path, str(save_dict['cur_num'])+'.jpg'), image[y1:y2, x1: x2, :])
        save_dict['cur_num'] += 1

        # if accident == ACCIENT_SURE or accident == ACCIENT_WARMING:
        if True:
            if int(speed) > 5:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(image, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
        # tf = max(tl - 1, 1)  # font thickness
        # t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(image, c1, c2, color, 2, cv2.LINE_AA)  # filled
            # 打印速度等信息
            # cv2.putText(image, '{}-{}-{}'.format(track_id, int(speed), int(speed_a)), (c1[0], c1[1] + 10), 0, 1,
            #             [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            # cv2.putText(image, '{}-{}'.format(track_id, int(speed)), (c1[0], c1[1] + 10), 0, 1,
            #             [225, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(image, '{}'.format(int(speed)), (c1[0], c1[1] + 10), 0, 1,
                        [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        if len(points) < point_fps:
            start = 0
        else:
            start = len(points) - point_fps
        last_point = points[start]
        for point in points[start:-1]:
            # cv2.line(image, last_point, point, (255, 0, 0), 5)
            last_point = point

    return image

'''
speed_skip:算加速度时，跟多少帧前进行比较
stop_speed：小于这个数，算停车了
stop_round：连续多少帧停车开始判断是不是车祸
max_speed: 从这个车出现到现在，最大速度超过了这个值，才进行判断，否则就认为是一直停着
speed_a_times: 如果已经确定车停了好久，需要看车停之前speed_a_times帧的加速度有没有减小很快的
min_speed_a: 判断减速的加速度时，小于这个值，认定是异常停车
'''
def update_tracker(target_detector, image, cars, speed_skip, stop_speed=1, stop_round=24, max_speed=10,
                   speed_a_times=10, min_speed_a=-3, context=None, last_bboxes=None, last_outputs=None, tt_dict=None):

        # new_faces = []
        t1 = time.time()
        if last_bboxes is None:
            _, bboxes = target_detector.detect(image)
        else:
            bboxes = last_bboxes
        tt_dict['1'] += time.time() - t1

        bbox_xywh = []
        confs = []
        bboxes2draw = []
        # bboxes = []
        ids = []
        outputs = None
        if len(bboxes):
            # Adapt detections to deep sort input format
            t1 = time.time()
            for x1, y1, x2, y2, _, conf in bboxes:
                
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # Pass detections to deepsort
            if last_outputs is None:
                outputs = deepsort.update(xywhs, confss, image, tt_dict)
            else:
                outputs = last_outputs
            tt_dict['2'] += time.time() - t1
            t1 = time.time()
            for value in list(outputs):
                x1,y1,x2,y2,track_id = value
                track_id = str(track_id)
                ids.append(track_id)
                # bboxes.append((x1, y1, x2, y2))
                if track_id in cars:
                    car = cars[track_id]
                else:
                    # stop_calc:[{停止的x,y,w,h,停止了t帧}]
                    car = {'track_id': track_id, 'distances': [], 'points': [], 'lastpoint': None, 'speed': 0, 'speed_a': 0,
                           'max_speed': 0, 'last_not_stop_index': -1, 'last_frame_stop': False, 'stop': False, 'accident': ACCIENT_NORMAL, 'speeds': [], 'speeds_a': [], 'angles': [], 'angles_a': [],
                           'stop_count': 0, 'stop_calc': []}
                    # cars[track_id] = car
                center = (int((x2 + x1) / 2), int((y2 + y1)/2))
                car['points'].append(center)
                lastpoint = car['lastpoint']
                lastdistence = 0
                if lastpoint is not None:
                    lastdistence = euclidean_distance(center, lastpoint)
                    car['distances'].append(lastdistence)
                car['lastdistence'] = lastdistence
                car['lastpoint'] = (center)
                car['location'] = (x1,y1,x2,y2)

                if len(car['points']) < speed_skip:
                    speed_frame = 0
                else:
                    speed_frame = len(car['points']) - speed_skip
                # 计算速度
                speed = euclidean_distance(car['points'][speed_frame], car['points'][-1]) * 1000 / (((x2 - x1)*(y2 - y1))**0.5) / 12
                if speed > 10:
                    if ran() > 0.7:
                        speed = ran(car['speed'] - 1, car['speed'] + 1)
                        speed = 10 if speed > 10 else speed
                        speed = ran(2, 5) if speed < 0 else speed
                    else:
                        speed = car['speed']
                # speed = euclidean_distance(car['points'][speed_frame], car['points'][-1])
                angle = calc_angle(car['points'][speed_frame][0], car['points'][speed_frame][1], car['points'][-1][0], car['points'][-1][1])
                car['speed'] = speed
                car['speeds'].append(int(speed))
                car['angle'] = angle
                car['angles'].append(int(angle))
                speed_a = car['speeds'][-1] - car['speeds'][speed_frame]
                car['speed_a'] = speed_a
                car['speeds_a'].append(int(speed_a))
                angle_a = abs(car['angles'][-1] - car['angles'][speed_frame])
                car['angle_a'] = angle_a
                car['angles_a'].append(int(angle_a))
                if speed > car['max_speed']:
                    car['max_speed'] = speed
                if car['stop']:
                    if speed > stop_speed:
                        car['stop'] = False
                        # 完成一次停车，记录一下时间
                        car['stop_calc'][-1][4] = car['stop_count']
                        print("car['stop_count']:", car['stop_count'], speed, car['points'][speed_frame], car['points'][-1], x2 - x1, y2 - y1)
                        car['stop_count'] = 0
                        car['last_not_stop_index'] = len(car['speeds_a'])-1
                    else:
                        car['stop'] = True
                        car['stop_count'] += 1
                        # 判断是不是车祸
                        if context['stop_mean'] != 0 and context['stop_variance'] != 0:
                            if car['stop_count'] > context['stop_mean'] + 3 * context['stop_variance']:
                                print(
                                    "ACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SUREACCIENT_SURE")
                                car['accident'] = ACCIENT_SURE
                            elif car['stop_count'] > context['stop_mean'] + 2 * context['stop_variance']:
                                car['accident'] = ACCIENT_WARMING
                                print(
                                    "ACCIENT_WARMINGACCIENT_WARMINGACCIENT_WARMINGACCIENT_WARMINGACCIENT_WARMING")
                else:
                    # 最大速度大于10 且最小速度小于5
                    if speed > stop_speed:
                        car['stop'] = False
                        car['stop_count'] = 0
                        car['last_not_stop_index'] = len(car['speeds_a'])-1
                    else:
                        if car['max_speed'] > max_speed:
                            car['stop_calc'].append([x1, y1 ,x2 ,y2 ,1])
                            car['stop'] = True
                            # # 如果停了很久，看停之前若干帧有没有加速度减小很快的
                            # if car['stop_count'] > stop_round:
                            #     last_not_stop_index = car['last_not_stop_index']
                            #     if last_not_stop_index < speed_a_times:
                            #         start_index = 0
                            #     else:
                            #         start_index = last_not_stop_index - speed_a_times
                            #     min_speed_change = np.min(car['speeds_a'][start_index:last_not_stop_index])
                            #     if min_speed_change < min_speed_a:
                            #         print('stop:', track_id, speed, min_speed_a, car['stop_count'])
                            #         car['stop'] = True
                            car['stop_count'] = 1
                        else:
                            car['stop_count'] = 0
                cars[track_id] = car
                # if track_id=='51':
                bboxes2draw.append(car)
                # if track_id == 1:
                #     print('car1111111111111:', speed, car['stop'])
        tt_dict['3'] += time.time() - t1
        # t1 = time.time()
        image = plot_bboxes(image, bboxes2draw)
        # tt_dict['3'] += time.time() - t1
        return image, ids, bboxes, cars, outputs

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def calc_angle(x1, y1, x2, y2):
    if x1 == x2:
        return 90
    if y1 == y2:
        return 180
    k = -(y2 - y1) / (x2 - x1)
    # 求反正切，再将得到的弧度转换为度
    result = np.arctan(k) * 57.29577
    # 234象限
    if x1 > x2 and y1 > y2:
        result += 180
    elif x1 > x2 and y1 < y2:
        result += 180
    elif x1 < x2 and y1 < y2:
        result += 360
    # print("直线倾斜角度为：" + str(result) + "度")
    return result

def ran(a=0, b=1):
    return np.random.rand() * (b - a) + a