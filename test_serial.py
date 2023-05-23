import time
import keyboard
import threading
import subprocess
import random
import serial
import signal

port = "/dev/ttyTHS0"
baud = 115200
exitThread = False
line = []


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
        # Data1
        packet.append(0x48)
        packet.append(0x4F)
        # Data2
        packet.append(0x48)
        packet.append(0x4F)
        # Data3
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
        packet.append(0x00)
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
        packet.append(0x00)
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
    
    
if __name__ == "__main__":
    ser = serial.Serial(port, baud, timeout=1)
    thread = threading.Thread(target=readThread, args=(ser,))
    thread.start()
    
    while True:
        # situa
        time.sleep(10)
        ser.write(packing_data(0x02, 0))
        # ser.write(packing_data(0x03, 0))
        print("Loop2")
        
