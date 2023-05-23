from caption_utils import *
import torch
import os
import numpy as np
import pickle5 as pickle
import time
#import keyboard
import threading
import subprocess
import random
#import serial
import signal

port = "/dev/ttyTHS0"
baud = 115200
exitThread = False
line = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

look_up_tables_path = './region_dict.pkl'
look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))
idx_to_token = look_up_tables['idx_to_token']

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
rcnn = create_model(num_classes=6).to(device)
rcnn.load_state_dict(torch.load('./model_100.pth'))
rcnn.eval()


describer = describer_v2(is_training=False).to(device)
tmp_captions = {
    0: "A person is approaching",
    1: "A delivery man is approaching",
    # 2: "A person is approaching with cart",
    # 3: "A delivery man is approaching",
    # 4: "A delivery man is approaching with box",
    # 5: "A delivery man is approaching with cart",
    # 6: "A person is standing"
}

def handler(signum, frame):
    print("End")
    exitThread = True


def packing_data(req, data):
    packet = []
    # STX
    packet.append(0x02)
    if req == 0x01:
        # CMD
        packet.append(0x02)
        # Data1 Width
        packet.append(0x48)
        packet.append(0x4F)
        # Data2 Height
        packet.append(0x48)
        packet.append(0x4F)
        # Data3 Depth
        packet.append(0x48)
        packet.append(0x4F)
        # ETX
        packet.append(0x03)
        # SUM
        packet.append(sum(packet).to_bytes(2, byteorder="little")[0])
    elif req == 0x02:
        # CMD
        packet.append(0x03)
        # Data1
        packet.append(0x01)
        packet.append(0x00) # for veiw defult is 0x00
        # Data2
        packet.append(0x00)
        packet.append(0x00)
        # Data3
        packet.append(0x00)
        packet.append(0x00)
        # ETX
        packet.append(0x03)
        # SUM
        packet.append(sum(packet).to_bytes(2, byteorder="little")[0])
    elif req == 0x03:
        # CMD
        packet.append(0x03)
        # Data1
        packet.append(0x02)
        packet.append(0x00) # for veiw defult is 0x00
        # Data2
        packet.append(0x00)
        packet.append(0x00)
        # Data3
        packet.append(0x00)
        packet.append(0x00)
        # ETX
        packet.append(0x03)
        # SUM
        packet.append(sum(packet).to_bytes(2, byteorder="little")[0])
    return packet


def readThread(ser):
    global line
    
    while not exitThread:
        for c in ser.read():
            line.append(c)
            if len(line) >= 4:
                print(line)
                _sum = sum(line[:-1])
                if line[1] == 0x01 and _sum == line[-1]:
                    print("req")
                    # volume
                    ser.write(packing_data(0x01, 0))
                del line[:]


def check_situation():
    subprocess.call("v4l2-ctl --device /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=MJPG --stream-mmap --stream-to=frame.jpg --stream-count=1", shell=True)
    
    # Faster RCNN
    path_image = './frame.jpg'

    start_time = time.time()
    image, image_size, raw_img = load_image(path_image)
    try:
        print("load ---{}s seconds---".format(time.time()-start_time))
        rcnn_preds = rcnn(image)[0]
        print("rcnn ---{}s seconds---".format(time.time()-start_time))
        dre_boxes = dre(rcnn_preds, image_size, score_thresh=0.7, margin=0.0)  # DRE
        print("dre  ---{}s seconds---".format(time.time()-start_time))
        features = rcnn.backbone(image)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        obj_features = roi_pooler(features, [torch.stack(dre_boxes)], [image_size])

        pred_captions = describer(obj_features)  # captioning
        print("caps  ---{}s seconds---".format(time.time()-start_time))
        # distances = cal_distance(dre_boxes)  # Cal distances
        # print("dist  ---{}s seconds---".format(time.time()-start_time))

        d_cnt = 0
        c_cnt = 0

        for caption_idx in pred_captions.tolist():
            if caption_idx == 0: d_cnt += 1
            elif caption_idx == 1: c_cnt += 1

        if d_cnt >= c_cnt:
            return 0x01
        else:
            return 0x02
    except:
        print('object not found')

    
if __name__ == "__main__":
    ser = serial.Serial(port, baud, timeout=1)
    thread = threading.Thread(target=readThread, args=(ser,))
    thread.start()
    
    while True:
        # situa
        time.sleep(10)
        ser.write(packing_data(check_situation(), 0))
        # ser.write(packing_data(0x03, 0))
        print("Loop")
        
