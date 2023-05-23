from caption_utils import *
import torch
import os
import numpy as np
import pickle5 as pickle
import time
import keyboard
import threading
import subprocess
import random
import serial
import signal
import struct


port_comm = "/dev/ttyTHS0"
baud_comm = 115200

port_tof = "/dev/ttyUSB0"
baud_tof = 921600

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
        w, h, d = data
        data_w = list(struct.pack("!H", int(math.ceil(w))))
        data_h = list(struct.pack("!H", int(math.ceil(h))))
        data_d = list(struct.pack("!H", int(math.ceil(d))))
        # CMD
        packet.append(0x02)
        # Data1 (W)
        packet.append(data_w[0])
        packet.append(data_w[1])
        # Data2 (H)
        packet.append(data_h[0])
        packet.append(data_h[1])
        # Data3 (D)
        packet.append(data_d[0])
        packet.append(data_d[1])
        # ETX
        packet.append(0x03)
        # SUM
        packet.append(sum(packet).to_bytes(2, byteorder="little")[0])
    elif req == 0x02:
        # CMD
        packet.append(0x03)
        # Data1
        packet.append(0x01)
        packet.append(0x4D) # for veiw defult is 0x00
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
        packet.append(0x4E) # for veiw defult is 0x00
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
                _sum = sum(line[:-1])
                if line[1] == 0x01 and _sum == line[-1]:
                    print(f"req : {line}")
                    # volume
                    ser.write(packing_data(0x01, check_volumne()))
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
            return 0x02
        else:
            return 0x03
    except:
        print('object not found')
    
    
def tofThread(ser):
    global line_tof
    
    cos = [0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427,
           0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427,
           0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.96043427, 0.98420279, 0.98772502,
           0.97799505, 0.98067089, 0.97934846, 0.98136816, 0.98381701, 0.98564392, 0.98634829, 0.98985041,
           0.97976567, 0.98385193, 0.9873717, 0.98687746, 0.99148615, 0.99091653, 0.99323444, 0.9931273,
           0.97983526, 0.98578472, 0.9904898, 0.99162865, 0.99427209, 0.99775868, 0.99534787, 0.99416472]
    
    size_buff = 0
    zeros = np.zeros(48)
    zerosList = []
    serStack = []
    dataStack = []
    maxtof = 999
    while not exitThread:
    
        if ser.readable():
            time.sleep(0.01)
            data = ser.read_until(b'\x57\x01')
            if len(data) == 400:
                byte_array = bytearray(data[7:-9])
                byte_array = np.array(byte_array)
                byte_array = byte_array.reshape(8, 8, 6)
                byte_array = byte_array[:, :, 0:3]
                byte_array = byte_array[:, :, ::-1]
                dataList = []
                for i in range(8):
                    for j in range(8):
                        dis = byte_array[i, j, 0] * (16 ** 4) + \
                              byte_array[i, j, 1] * (16 ** 2) + \
                              byte_array[i, j, 2]
                        dataList.append(dis)
                serStack.append(dataList)
                size_buff += 1
        if size_buff >= 30:
            for arr in serStack:
                a = arr[:48]
                maybe_height = 0
                listCount = 0
                rowList =[]
                receivList =[]
                dataCount = 0
                for i in range(48):
                    maybe_height = a[i]*cos[i]
                    if maybe_height > 230000:
                        maybe_height = maybe_height - 20000

                    maybe_height = 1.10744 * maybe_height - 24856.64786
                    if maybe_height > 700000 or maybe_height < 0:
                        maybe_height = 0
                        zeros[i] += 1
                    listCount += 1
                    rowList.append(maybe_height)
                    if listCount > 7:
                        receivList.append(rowList)
                        rowList = []
                        listCount = 0
                        dataCount += 1
                        if dataCount > 5:
                            dataStack.append(receivList)
                            receivList = []
                            dataCount = 0
                zerosList.append(zeros)
                zeros = np.zeros(48)

        if size_buff == 30:
            zeroStack = np.zeros(48)
            for i in zerosList:
                zeroStack += i

            stack2 = np.zeros(48)
            for i in dataStack:
                stack1 = i[0] + i[1] + i[2] + i[3] + i[4] + i[5]
                for j in range(len(stack1)):
                    stack2[j] = stack2[j] + stack1[j]

            for i in range(len(stack2)):
                stack2[i] = stack2[i] / 10000

            for i in range(len(stack2)):
                nonzero = 30 - zeroStack[i]
                if nonzero > 5:
                    if (maxtof > stack2[i] / nonzero):
                        maxtof = stack2[i] / nonzero

            maxtof = 78 - maxtof
            line_tof = maxtof

            dataStack = []
            zerosList = []
            zeros = np.zeros(48)
            size_buff = 0
            dataCount = 0
            listCount = 0
            maybe_height = 0


def cal_area(bg_image, cap_image, boxHeight):
    img_001h = cv2.resize(bg_image, dsize=(640, 480),
                          interpolation=cv2.INTER_AREA)
    img_002h = cv2.resize(cap_image, dsize=(640, 480),
                          interpolation=cv2.INTER_AREA)

    img_001h = cv2.cvtColor(img_001h, cv2.COLOR_BGR2HSV)
    img_002h = cv2.cvtColor(img_002h, cv2.COLOR_BGR2HSV)

    img_001h = cv2.normalize(img_001h, None, 0, 255, cv2.NORM_MINMAX)
    img_002h = cv2.normalize(img_002h, None, 0, 255, cv2.NORM_MINMAX)

    img_cha = cv2.absdiff(img_001h, img_002h)
    img_chaH, img_chaS, imghs = cv2.split(img_cha)

    imghs = cv2.add(img_chaH, img_chaS)
    imghs = cv2.normalize(imghs, None, 0, 255, cv2.NORM_MINMAX)

    edgep = cv2.edgePreservingFilter(imghs, flags=1, sigma_s=45, sigma_r=0.2)
    medianp = cv2.medianBlur(edgep, 3)
    cannyp = cv2.Canny(medianp, 50, 20, True, apertureSize=3)

    contour_in = copy.deepcopy(cannyp)
    (cnts, _) = cv2.findContours(contour_in.copy(), cv2.RETR_LIST,
                                 cv2.CHAIN_APPROX_NONE)

    thresLen = 150
    filteredCont = []

    for cnt in cnts:
        lens = cv2.arcLength(cnt, False)
        if lens >= thresLen:
            approx = cv2.approxPolyDP(cnt, 0.01 * lens, False)
            filteredCont.append(approx)

    bestRect = [400, 224, 0, 0]

    for filtercnt in filteredCont:
        rect = cv2.boundingRect(filtercnt)

        if bestRect[0] > rect[0]:
            bestRect[0] = rect[0]

        if (bestRect[0] + bestRect[2]) < (rect[0] + rect[2]):
            bestRect[2] = rect[0] + rect[2] - bestRect[0]

        if bestRect[1] > rect[1]:
            bestRect[1] = rect[1]

        if (bestRect[1] + bestRect[3]) < (rect[1] + rect[3]):
            if (rect[1] + rect[3]) > 165:
                bestRect[3] = 165 - bestRect[1]
            else:
                bestRect[3] = (rect[1] + rect[3]) - bestRect[1]

    height = boxHeight

    unit_cm = 0.0214 * height * height + 0.4469 * height + 38.112

    box_width = (bestRect[2] / unit_cm) * 10.5 * 10
    box_height = (bestRect[3] / unit_cm) * 10.5 * 10


    return box_width, box_height


def capture_background():
    subprocess.call("v4l2-ctl --device /dev/video1 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-to=inner.jpg --stream-count=1", shell=True)
    
    path_image = './inner.jpg'
    CameraMtx = np.array([[490.7367888, 0.0, 582.5049137],
                      [0.0, 490.8820335, 363.6448768],
                      [0.0, 0.0, 1.0]])

    distCoeffs = np.array([-0.249900783, 0.081937333, 0.000197873, 0.00025773, -0.013393505])

    NewCameramtx = np.array([[170.3414307, 0.0, 942.6934251],
                         [0.0, 303.3730469, 370.1711453],
                         [0.0, 0.0, 1.0]])

    Roi = np.array([775, 205, 320, 320])

    mapx, mapy = cv2.initUndistortRectifyMap(CameraMtx, distCoeffs, None, 
    NewCameramtx, (1280, 720), 5)

    frame = cv2.imread(path_image)
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = Roi
    cap_image = frame[y:y + h, x:x + w]
    cv2.imwrite("./inner_b.jpg", cap_image)


def check_volumne():
    subprocess.call("v4l2-ctl --device /dev/video1 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-to=inner.jpg --stream-count=1", shell=True)
    
    path_image = './inner.jpg'
    CameraMtx = np.array([[490.7367888, 0.0, 582.5049137],
                      [0.0, 490.8820335, 363.6448768],
                      [0.0, 0.0, 1.0]])

    distCoeffs = np.array([-0.249900783, 0.081937333, 0.000197873, 0.00025773, -0.013393505])

    NewCameramtx = np.array([[170.3414307, 0.0, 942.6934251],
                         [0.0, 303.3730469, 370.1711453],
                         [0.0, 0.0, 1.0]])

    Roi = np.array([775, 205, 320, 320])

    mapx, mapy = cv2.initUndistortRectifyMap(CameraMtx, distCoeffs, None, 
    NewCameramtx, (1280, 720), 5)

    frame = cv2.imread(path_image)
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = Roi
    cap_image = frame[y:y + h, x:x + w]
    cv2.imwrite("./inner_1.jpg", cap_image)
    
    back_image = cv2.imread("./inner_b.jpg")
    
    d = line_tof
    w, h = cal_area(back_image, cap_image, d)
    
    return (w, h, d)

    
if __name__ == "__main__":
    ser_comm = serial.Serial(port_comm, baud_comm, timeout=1)
    thread_comm = threading.Thread(target=readThread, args=(ser_comm,))
    thread_comm.start()
    
    ser_tof = serial.Serial(port_tof, baud_tof, timeout=1)
    thread_tof = threading.Thread(target=tofThread, args=(ser_tof,))
    thread_tof.start()
    
    while True:
        # situa
        time.sleep(2)
        capture_background()
        
        time.sleep(8)
        ser_comm.write(packing_data(check_situation(), 0))
        print(check_situation())
        # ser_comm.write(packing_data(0x01, check_volumne()))
        
        print("Loop")
        
