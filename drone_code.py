from djitellopy import tello
import cv2
import numpy as np
from time import sleep, time

me = tello.Tello()
me.connect()
sleep(5)
me.streamon()
cap = me.get_frame_read()

state = 1
start_time = time()
kp = 0.5
target_center = (320, 240)

def log_state(selected_state):
    with open("state_log.txt", "a") as file:
        file.write(f"Selected State: {selected_state}\n")

try:
    print("Start")
    me.takeoff()

    while True:
        frame = cap.frame
        qr_detector = cv2.QRCodeDetector()
        data, bbox, _ = qr_detector.detectAndDecode(frame)

        elapsed_time = time() - start_time
        if elapsed_time > 20:
            print("Timeout")
            me.land()
            break

        user_input = input()
        if user_input.isdigit():
            new_state = int(user_input)
            if 1 <= new_state <= 7:
                state = new_state
                log_state(state)
                
        if state == 1:
            print("State 1: Search target")
            x_vel = np.sin(elapsed_time)*10
            y_vel = np.sin(elapsed_time)*10
            me.send_rc_control(y_vel, x_vel, 0, 0)

        elif state == 2:
            print("State 2: Center target")

            if bbox is not None:
                bbox_center_x = int((bbox[0][0][0] + bbox[0][2][0]) / 2)
                bbox_center_y = int((bbox[0][0][1] + bbox[0][2][1]) / 2)
                error_x = target_center[0] - bbox_center_x
                error_y = target_center[1] - bbox_center_y

                control_x = int(kp * error_x)
                control_y = int(kp * error_y)
                vel_x = me.get_speed_x()
                vel_y = me.get_speed_y()

                me.send_rc_control(
                    int(np.clip((control_x - vel_x)/-10, -10, 10)),
                    int(np.clip((control_y - vel_y)/-10, -10, 10)),
                    0, 0
                )

        elif state == 3:
            print("State 3: Move forward")
            me.send_rc_control(0, 5, 0, 0)

        elif state == 4:
            print("State 4: Move backwards")
            me.send_rc_control(0, -5, 0, 0)

        elif state == 5:
            print("State 5: Landing target safely")
            me.land()
            break

        elif state == 6:
            print("State 6: Land backwards")
            me.send_rc_control(0, -5, 0, 0)
            me.land()

        elif state == 7:
            print("State 7: Land forward")
            me.send_rc_control(0, 5, 0, 0)
            me.land()

        sleep(0.1)

except Exception as e:
    print(f"Error: {e}")

finally:
    me.streamoff()
