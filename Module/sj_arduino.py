import prj_info
import sys
sys.path.append(prj_info.module_path)
from sj_file_system import str_join

import serial
import serial.tools.list_ports

def connected_ports():
    myports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
    return myports


def find_arduino_automatically():
    port_list = connected_ports()

    for port in port_list:
        if "Arduino" in str_join(port):
            print("find!")
            return port
    
    return None

class Comm_arduino:
    comm_arduino = None

    def __init__(self, port, baudrate = 9600):
        # ex: arduino = serial.Serial(port="/dev/cu.usbmodem14201", baudrate=9600)

        self.comm_arduino = serial.Serial(port=port, baudrate=baudrate)

    def write_read(self, input):
        self.comm_arduino.write(bytes(input, "utf-8"))

    def close(self):
        self.comm_arduino.close()

if __name__ == "__main__":
    arduino = Comm_arduino(port = "/dev/cu.usbmodem14201", baudrate = 9600)
    arduino.write_read("a")
