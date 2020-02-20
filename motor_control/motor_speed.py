import sys
import os
def motor_speed(data):
    y_center = (data[4] + data[6]) / 2
#    y_center = (data.data[4] + data.data[6]) / 2
    width = 378
    return min(100, 100 * (width - y_center) / (width * 0.4))

def main():
    data = [0,0,0,0,300,0,300]
    while True:
        cur_speed = motor_speed(data)
        print('speed: ', cur_speed)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:
        print('lol')