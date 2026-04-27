from Interpreter import Visualnterpreter
from DrawUtils import SpeechBubble
import cv2
import datetime
import json
import math
import mediapipe as mp
import os
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark
import numpy as np
import requests

FONT_SIZE = 10
FONT_WEIGHT = 10
MIN_CHARS_PER_LINE = 15
MAX_LINES_OF_CODE = 2
HORIZONTAL_MARGIN = 10

COMMAND_DELAY = 0

# Threshold can be updated by clicking the video stream
# Use the g command to view and test the updated thresholds
THRESHOLD_DUCK_Y = 250 # Full body: ~200, Office desk: ~400
THRESHOLD_EDGE = 430

SHOW_GRID_LINES = False

OFFSETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offsets.json')

def load_offsets():
    global THRESHOLD_DUCK_Y, THRESHOLD_EDGE
    if os.path.exists(OFFSETS_FILE):
        try:
            with open(OFFSETS_FILE, 'r') as f:
                data = json.load(f)
            THRESHOLD_DUCK_Y = data.get('THRESHOLD_DUCK_Y', THRESHOLD_DUCK_Y)
            THRESHOLD_EDGE = data.get('THRESHOLD_EDGE', THRESHOLD_EDGE)
        except (json.JSONDecodeError, IOError):
            pass

def save_offsets():
    with open(OFFSETS_FILE, 'w') as f:
        json.dump({'THRESHOLD_DUCK_Y': THRESHOLD_DUCK_Y, 'THRESHOLD_EDGE': THRESHOLD_EDGE}, f)

load_offsets()

COMPETITION_MODE = False
COMPETITION_WORD = 'kode24'
CAMERA_INDEX = 0

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotator', 'pose_landmarker.task')

_FACE_LANDMARK_INDICES = {
    mp_vision.PoseLandmark.LEFT_EYE.value,
    mp_vision.PoseLandmark.RIGHT_EYE.value,
    mp_vision.PoseLandmark.LEFT_EYE_INNER.value,
    mp_vision.PoseLandmark.RIGHT_EYE_INNER.value,
    mp_vision.PoseLandmark.LEFT_EAR.value,
    mp_vision.PoseLandmark.RIGHT_EAR.value,
    mp_vision.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_vision.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_vision.PoseLandmark.NOSE.value,
    mp_vision.PoseLandmark.MOUTH_LEFT.value,
    mp_vision.PoseLandmark.MOUTH_RIGHT.value,
}
PoseLandmark = mp_vision.PoseLandmark

class PoseDetector():
    def __init__(self, detectionCon=0.5, trackCon=0.5):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)
        self._connections = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
        self.detection_result = None

    def process(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(imgRGB))
        self.detection_result = self.pose.detect(mp_image)

        if self.detection_result.pose_landmarks and draw:
            pose_landmarks = self.detection_result.pose_landmarks[0]
            # Hide face landmarks by zeroing visibility
            filtered = [
                NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z,
                                   visibility=0.0 if i in _FACE_LANDMARK_INDICES else lm.visibility)
                for i, lm in enumerate(pose_landmarks)
            ]
            mp_vision.drawing_utils.draw_landmarks(img, filtered, self._connections)

        return img

    def find_pixel_positions(self, img):
        self.landmark_list = []
        if self.detection_result and self.detection_result.pose_landmarks:
            pose_landmarks = self.detection_result.pose_landmarks[0]
            # Determining the pixel position of the landmarks
            h, w, _ = img.shape
            for id, lm in enumerate(pose_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id, cx, cy])
        return self.landmark_list

    def find_angle(self, p1, p2, p3):
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]

        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle

        return angle

    def find_length(self, p1, p2):
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance  

def get_text_width(text, font_face, font_scale, font_line_thickness):
    ((txt_w, _), _) = cv2.getTextSize(text, font_face, font_scale, font_line_thickness)
    return txt_w

def draw_white_apha_box(img, x, y, h, w):
    # Crop the sub-rect from the image
    # https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
    sub_img = img[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    # Put the image back to its position
    img[y:y+h, x:x+w] = res

def main():
    global CAMERA_INDEX
    global SHOW_GRID_LINES

    detector = PoseDetector()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    show_code_lines = True
    SHOW_GRID_LINES = False

    last_command = ''
    same_command_count = 0
    code = ''
    lines_of_code = []

    clap_count = 0
    clap_stage = ''
    clap_closing_timeframe = 0
    clap_display_for_frames = 0
    clap_print1 = 0
    clap_print2 = 0

    print_lock = 0
    pause = False
    execute_code = False
    reload_code = False
    code_output = ''
    interpreter_finished_debug_and_print = False
    interpreter_paused = False
    interpreter_stopped = False
    interpreter_error = False
    interpreter_error_line = 0
    interpreter_error_char = 0
    step_forward = False
    step_back = False
    interpreter = Visualnterpreter()
    speech_bubble = SpeechBubble()

    nova_start_time = None
    nova_end_time = None
    total_time = ''
    nova_started = False
    nova_printed = False
    fullscreen = False

    # Menu
    print(' ')
    print('        c: Toggle code view')
    print('        g: Toggle grid')
    print('        p: Pause')
    print('backspace: Delete single character')
    print('   delete: Clear code')
    print('      F11: Toggle fullscreen')
    print(' ')

    while cap.isOpened():
        ready, flipped_frame = cap.read()

        if ready:    
            frame = cv2.flip(flipped_frame, 1)
            annotated_frame = detector.process(frame, draw=not pause)
            landmarks = detector.find_pixel_positions(frame)

            h, w, _ = frame.shape
            THRESHOLD_LEFT_X = 640 - THRESHOLD_EDGE
            THRESHOLD_RIGHT_X = THRESHOLD_EDGE

            if SHOW_GRID_LINES:
                cv2.line(frame, (THRESHOLD_LEFT_X, 0), (THRESHOLD_LEFT_X, h), (111,111,111), 2)
                cv2.line(frame, (THRESHOLD_RIGHT_X, 0), (THRESHOLD_RIGHT_X, h), (111,111,111), 2)
                cv2.line(frame, (0, THRESHOLD_DUCK_Y), (w, THRESHOLD_DUCK_Y), (111,111,111), 2)
            # w = 640
            # h = 480
            if execute_code:
                finished = False
                interpreter.debug_lines_of_code(frame, (int(HORIZONTAL_MARGIN / 2)))
                if not interpreter_paused and not interpreter_stopped and not interpreter_finished_debug_and_print and not pause or (interpreter_paused and (step_forward or step_back)):
                    complete_outout = None
                    if interpreter_paused:
                        if step_forward:
                            step_forward = False
                            finished, remember, c, l, o = interpreter.step(single_step=True)
                        if step_back:
                            step_back = False
                            interpreter_finished_debug_and_print = False
                            finished, remember, c, l, complete_outout = interpreter.step_back()
                    else:
                        finished, remember, c, l, o = interpreter.step()
                    if o:
                        code_output += o

                    if complete_outout != None:
                        code_output = complete_outout
                        
                    if remember:
                        interpreter.history_append(code_output)

                    if finished:
                        # When resuming interpreting after stepping, make sure we not start from beginning after finishing
                        clap_count = 0
                            
                    if finished and not interpreter_finished_debug_and_print:
                        interpreter_finished_debug_and_print = True
                        interpreter_paused = True
                interpreter.print_cells(frame)
                if code_output == COMPETITION_WORD or COMPETITION_MODE == False:
                    interpreter.print_outout(frame, code_output, (0,255,0))
                else:
                    if interpreter_finished_debug_and_print:                        
                        interpreter.print_outout(frame, code_output, (0,0,255))
                    else:
                        interpreter.print_outout(frame, code_output, (255,255,255))

                if interpreter_error:
                    interpreter.highlight_debug_command(frame, interpreter_error_char, interpreter_error_line, (int(HORIZONTAL_MARGIN / 2)), (0, 0, 255))
                elif pause or (not finished and not interpreter_stopped and not interpreter_finished_debug_and_print):
                    interpreter.highlight_debug_command(frame, c, l, (int(HORIZONTAL_MARGIN / 2)))                        

            if show_code_lines:
                lines_of_code = []
                code_left_to_print = code.strip()
                while len(code_left_to_print) > 0:
                    if len(code_left_to_print) > MIN_CHARS_PER_LINE:
                        char_count = MIN_CHARS_PER_LINE
                        line_width = get_text_width(code_left_to_print[:char_count], cv2.FONT_HERSHEY_PLAIN, 2, 2)
                        while line_width < w - HORIZONTAL_MARGIN and char_count < len(code_left_to_print):
                            char_count += 1
                            line_width = get_text_width(code_left_to_print[:char_count], cv2.FONT_HERSHEY_PLAIN, 2, 2)
                        if line_width > w - HORIZONTAL_MARGIN:
                            char_count -= 1
                        if char_count > len(code_left_to_print):
                            char_count = len(code_left_to_print)
                        lines_of_code.append(code_left_to_print[:char_count].strip()) 
                        code_left_to_print = code_left_to_print[char_count:]
                    else:
                        lines_of_code.append(code_left_to_print.strip())
                        code_left_to_print = ''

                interpreter.input_code(lines_of_code)

                # If code is updated with [ or ] while running, we need to update jump map
                if reload_code:
                    reload_code = False                     
                    ok, (interpreter_error_line, interpreter_error_char) = interpreter.build_jumpmap()
                    if ok:
                        interpreter_error = False
                    interpreter_paused = not ok
                    if interpreter_paused:
                        interpreter_error = True

                if not execute_code:
                    interpreter.print_lines_of_code(frame, MAX_LINES_OF_CODE, (int(HORIZONTAL_MARGIN / 2)))

            if len(landmarks) and not pause:
                elbow_l = detector.find_angle(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST)
                elbow_r = detector.find_angle(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST)
                upper_arm_l = detector.find_length(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW)
                upper_arm_r = detector.find_length(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW)
                half_upper_arm = int((upper_arm_l + upper_arm_r) / 4)
                upper_arm = int((upper_arm_l + upper_arm_r) / 2)

                # Elbow positions
                elbow_left_straight = elbow_l > 130
                elbow_right_straight = elbow_r > 130
                elbows_straight = elbow_left_straight and elbow_right_straight
                # Arms horizontal, less then half an upper arm off
                left_arm_horizonal = abs(landmarks[PoseLandmark.LEFT_SHOULDER][2] - landmarks[PoseLandmark.LEFT_WRIST][2]) < half_upper_arm
                right_arm_horizonal = abs(landmarks[PoseLandmark.RIGHT_SHOULDER][2] - landmarks[PoseLandmark.RIGHT_WRIST][2]) < half_upper_arm

                # Arms out, printing
                if elbows_straight and left_arm_horizonal and right_arm_horizonal:
                    if last_command == '.':
                        same_command_count += 1
                    elif print_lock == 0: # Avoid triggering double .
                        print_lock = 1
                        last_command = '.'
                        code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        draw_white_apha_box(frame, 260, 95, 110, 120)
                        # Print . a litle higher than other commands
                        cv2.putText(frame, '.', (280+15, 200-30), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)

                # Stepping debugger forward/back
                elif interpreter_paused and ((elbow_left_straight and left_arm_horizonal) or (elbow_right_straight and right_arm_horizonal)):
                    # Remberer that left and right are mirrored
                    if elbow_left_straight and left_arm_horizonal and not (elbow_right_straight and right_arm_horizonal):
                        if last_command == '-->':
                            same_command_count += 1
                        elif (last_command == 'default' or last_command == ''):
                            last_command = '-->'
                            same_command_count = 0
                            step_forward = True
                    elif elbow_right_straight and right_arm_horizonal and not (elbow_left_straight and left_arm_horizonal):
                        if last_command == '<--':
                            same_command_count += 1
                        elif (last_command == 'default' or last_command == ''):
                            last_command = '<--'
                            same_command_count = 0
                            step_back = True

                # Double-up, not included in original spec
                elif landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm: 
                    if last_command == '++':
                        same_command_count += 1
                    elif last_command == '+': # Upgrading directly from + to ++, should yield a total of ++ not +++
                        last_command = '++'
                        if code.endswith('+++++'):
                            code += ' +'
                        else:
                            code += '+'
                    else:
                        last_command = '++'
                        if code.endswith('+++++'):
                            code += ' ++'
                        elif code.endswith('++++'):
                            code += '+ +'
                        else:
                            code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        draw_white_apha_box(frame, 145, 95, 110, 355)
                        cv2.putText(frame, '+', (140, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
                        cv2.putText(frame, '+', (380, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)                                
               
                # Hands up!
                elif landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm or landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm : 
                    if last_command == '+':
                        same_command_count += 1
                    elif last_command == '++':  # Do not unintentional trigger single +, if not lowering both arms exacly at the same time 
                        same_command_count += 1
                    else:
                        last_command = '+'
                        if code.endswith('+++++'):
                            code += ' ' + last_command
                        else:
                            code += last_command
                        same_command_count = 0

                    if same_command_count > COMMAND_DELAY:
                        if landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm: 
                            draw_white_apha_box(frame, 145, 95, 110, 120)
                            cv2.putText(frame, '+', (140, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT) 
                        if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm:
                            draw_white_apha_box(frame, 385, 95, 110, 120)
                            cv2.putText(frame, '+', (380, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
            
                # Duck, shoulders below threshold
                elif landmarks[PoseLandmark.LEFT_SHOULDER][2] > THRESHOLD_DUCK_Y and landmarks[PoseLandmark.RIGHT_SHOULDER][2] > THRESHOLD_DUCK_Y: 
                    if last_command == '-':
                        same_command_count += 1
                    else:
                        last_command = '-'
                        if code.endswith('-----'):
                            code += ' ' + last_command
                        else:
                            code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        draw_white_apha_box(frame, 260, 95, 110, 120)
                        cv2.putText(frame, '-', (280-20, 200-5), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                
                # Body to the left
                elif landmarks[PoseLandmark.LEFT_SHOULDER][1] < THRESHOLD_LEFT_X and landmarks[PoseLandmark.RIGHT_SHOULDER][1] < THRESHOLD_LEFT_X:
                    if last_command == '<' or last_command == '[':
                        same_command_count += 1
                    else:
                        last_command = '<'
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY and same_command_count < 30:
                        draw_white_apha_box(frame, 260, 95, 110, 120)
                        cv2.putText(frame, '<', (260+5, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                    if same_command_count > 30:
                        if last_command != '[':
                            last_command = '['
                            code += last_command
                            if execute_code == True and interpreter_stopped == False and interpreter_finished_debug_and_print == False:
                                reload_code = True
                        draw_white_apha_box(frame, 260, 95, 110, 120)
                        # [ is strangely large, print it a little smaller, and further up, than other commands
                        cv2.putText(frame, '[', (280+20, 200-25), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE - 5, (0,0,255), FONT_WEIGHT)   
                
                # Body to the right
                elif landmarks[PoseLandmark.LEFT_SHOULDER][1] > THRESHOLD_RIGHT_X and landmarks[PoseLandmark.RIGHT_SHOULDER][1] > THRESHOLD_RIGHT_X:
                    if last_command == '>' or last_command == ']':
                        same_command_count += 1
                    else:
                        last_command = '>'
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY and same_command_count < 30:
                        draw_white_apha_box(frame, 260, 95, 110, 120)
                        cv2.putText(frame, '>', (260+5, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                    if same_command_count > 30:
                        if last_command != ']':
                            last_command = ']'
                            code += last_command
                            if execute_code == True and interpreter_stopped == False and interpreter_finished_debug_and_print == False:
                                reload_code = True
                        draw_white_apha_box(frame, 260, 95, 110, 120)
                        # ] is strangely large, print it a little smaller, and further up, than other commands
                        cv2.putText(frame, ']', (280+20, 200-25), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE - 5, (0,0,255), FONT_WEIGHT)   

                # Facepalm (right handed)
                # Index finger horizontally between the outer eyes, above eyes, not too far above head
                elif landmarks[PoseLandmark.LEFT_INDEX][1] < landmarks[PoseLandmark.LEFT_EYE_OUTER][1] \
                        and landmarks[PoseLandmark.LEFT_INDEX][1] > landmarks[PoseLandmark.RIGHT_EYE_OUTER][1] \
                        and landmarks[PoseLandmark.LEFT_INDEX][2] < landmarks[PoseLandmark.NOSE][2] \
                        and landmarks[PoseLandmark.LEFT_INDEX][2] > landmarks[PoseLandmark.NOSE][2] - half_upper_arm:
                    if last_command == '⌫':
                        same_command_count += 1
                    else:
                        last_command = '⌫'
                        same_command_count = 0
                        if code[-1:] in ['[',']'] and execute_code == True and interpreter_stopped == False and interpreter_finished_debug_and_print == False:
                            reload_code = True                    
                        code = code[:-1]
                    if same_command_count > COMMAND_DELAY:
                        bubble_x = landmarks[PoseLandmark.MOUTH_RIGHT][1]
                        bubble_y = landmarks[PoseLandmark.MOUTH_RIGHT][2]
                        speech_bubble.draw(frame, bubble_x, bubble_y, half_upper_arm)
                else:
                    if last_command in ['<','>']:
                        code += last_command
                    last_command = ''
                    same_command_count = 0 

                    # Dummy command to identify default position with arms down
                    # Clapping should be performed while starting in this position. With wrists higher than elbows
                    shoulder_r = detector.find_angle(PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW)
                    shoulder_l = detector.find_angle(PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW)
                    if shoulder_r < 60 and shoulder_l < 60: # Arms facing downwards
                        # Remember that right and left are mirrored, because the image is flipped
                        if landmarks[PoseLandmark.LEFT_SHOULDER][1] < 500 and landmarks[PoseLandmark.RIGHT_SHOULDER][1] < 500: # not too far right
                            if landmarks[PoseLandmark.LEFT_SHOULDER][1] > 140 and landmarks[PoseLandmark.RIGHT_SHOULDER][1] > 140: # not too far left
                                last_command = 'default'
                                print_lock = 0 # Must return to default between each print command
                                
                    if clap_print1 == 1 or clap_print2 == 1:
                        if clap_display_for_frames > 0:
                            clap_display_for_frames -= 1
                            if clap_count >= 2:
                                draw_white_apha_box(frame, 120, 95, 110, 400)
                                cv2.putText(frame, 'Clap! Clap!', (150, 180), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), FONT_WEIGHT)
                            elif clap_count == 1:
                                draw_white_apha_box(frame, 200-10, 95, 110, 220)
                                cv2.putText(frame, 'Clap!', (225, 180), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), FONT_WEIGHT)
                        else:
                            if clap_count == 1:
                                nova_end_time = datetime.datetime.now()
                                if execute_code:
                                    if interpreter_finished_debug_and_print or interpreter_stopped:
                                        interpreter.input_code(lines_of_code)
                                        ok, (interpreter_error_line, interpreter_error_char) = interpreter.prepare_code()
                                        code_output = ''
                                        execute_code = True
                                        if not ok:
                                            interpreter_error = True
                                            interpreter_stopped = True
                                        else:
                                            interpreter_error = False
                                            interpreter_paused = False
                                            interpreter_stopped = False
                                            interpreter_finished_debug_and_print = False
                                else:
                                    if len(code) > 0:
                                        interpreter.input_code(lines_of_code)
                                        ok, (interpreter_error_line, interpreter_error_char) = interpreter.prepare_code()
                                        code_output = ''
                                        execute_code = True
                                        if not ok:
                                            interpreter_error = True
                                            interpreter_stopped = True
                                        else:                                                                                        
                                            interpreter_error = False
                                            interpreter_paused = False
                                            interpreter_stopped = False
                                            interpreter_finished_debug_and_print = False
                            clap_print1 = 0
                            clap_print2 = 0
                            clap_stage = ''
                            clap_count = 0

                    if last_command == 'default' or last_command == '':
                        if landmarks[PoseLandmark.LEFT_INDEX][1] > landmarks[PoseLandmark.LEFT_SHOULDER][1] and landmarks[PoseLandmark.RIGHT_INDEX][1] < landmarks[PoseLandmark.RIGHT_SHOULDER][1]:
                            if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                if landmarks[PoseLandmark.LEFT_SHOULDER][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_SHOULDER][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                    clap_stage = 'wide' 
                                    clap_closing_timeframe = 25    
                        if clap_stage == 'wide' and clap_closing_timeframe > 0 and abs(landmarks[PoseLandmark.LEFT_INDEX][1] - landmarks[PoseLandmark.RIGHT_INDEX][1]) < int(half_upper_arm):
                            if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                clap_stage = 'clap'
                                clap_count += 1
                    if clap_count >= 2:
                        if clap_print2 == 0:
                            clap_display_for_frames = 10
                            clap_print2 = 1
                            # If interpreter is running: stop it
                            # Otherwise clear code buffer
                            if execute_code:
                                execute_code = False
                                interpreter_paused = False
                            else:
                                code = ''
                                code_output = ''
                                nova_end_time = None                 
                    elif clap_count == 1:
                        if clap_print1 == 0:
                            clap_display_for_frames = 20
                            clap_print1 = 1
                            # Pause / resume debugger immediately, without waiting for potential second clap
                            if execute_code and not pause and (interpreter_paused or not interpreter_finished_debug_and_print):
                                if interpreter_paused:
                                    interpreter_paused = False
                                else:
                                    interpreter_paused = True
                    elif clap_stage == 'wide' and clap_closing_timeframe > 0:
                        clap_closing_timeframe -= 1

            if last_command not in ['default', '']:
                nova_end_time = None

            if COMPETITION_MODE:
                if nova_started or nova_printed:
                    if nova_printed:
                        time = total_time
                    else:
                        if nova_end_time == None:
                            time = datetime.datetime.now() - nova_start_time
                        else:
                            time = nova_end_time - nova_start_time

                    score_color = (0,255,0)
                    formatted_time = f"{time.seconds // 60}:{time.seconds % 60:02}"
                    offset = interpreter.get_text_width(formatted_time, cv2.FONT_HERSHEY_PLAIN, 2, 2) - 2
                    cv2.putText(frame, formatted_time, (8 * 79 - offset, 465), cv2.FONT_HERSHEY_PLAIN, 2, score_color, 2)



            cv2.namedWindow('BodyFuck', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('BodyFuck', on_mouse)
            cv2.imshow('BodyFuck', annotated_frame)

        key = cv2.waitKeyEx(1)

        if key == 27:  # 27 == ESC key
            break

        if key != -1:
            if key == ord('c') or key == ord('C'): #Toggle code view
                show_code_lines = not show_code_lines
            elif key == ord('g') or key == ord('G'): #Toggle grid 
                SHOW_GRID_LINES = not SHOW_GRID_LINES
            elif key == 8: #Backspace
                if len(code) > 0 and code[-1] in ['[', ']']:
                    reload_code = True
                code = code[:-1]
                if code == '':
                    nova_end_time = None
            elif key == 3014656 or key == 2555904: #Clear code (delete key or right arrow / clicker)
                code = ''
                code_output = ''
                execute_code = False
                interpreter_paused = False
                nova_end_time = None
                # Make sure cells at the bottom of the screen is hidden
                ok, (interpreter_error_line, interpreter_error_char) = interpreter.prepare_code()
            elif key == ord('p') or key == ord('P'): #Pause
                pause = not pause
            elif key == 7995392: #F11
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty("BodyFuck", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty("BodyFuck", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if len(code) == 0:
            nova_started = False
            nova_printed = False
        elif nova_started == False:
            nova_started = True
            nova_printed = False
            nova_start_time = datetime.datetime.now()
            nova_end_time = None

        if code_output == COMPETITION_WORD and nova_printed == False: 
            if nova_end_time == None:
                total_time = datetime.datetime.now() - nova_start_time
            else:
                total_time = nova_end_time - nova_start_time
            time_spent_seconds = int(total_time.total_seconds())

            nova_printed = True
            try:
                resp = requests.post(
                    "http://127.0.0.1:3000/submit",
                    json={"time": time_spent_seconds},
                    timeout=3.0
                )
                resp.raise_for_status()
                print("Posted score, server replied:", resp.json())
            except Exception as e:
                print("Error posting score:", e)   

    cap.release()
    cv2.destroyAllWindows()

def on_mouse(event, x, y, flags, param):
    global THRESHOLD_EDGE, THRESHOLD_DUCK_Y
    if event == cv2.EVENT_LBUTTONDOWN and SHOW_GRID_LINES:
        if x > (640/2):
            THRESHOLD_EDGE = x
        else:
            THRESHOLD_EDGE = 640 - x
        THRESHOLD_DUCK_Y = y
        save_offsets()
        return
    
if __name__ == "__main__":
    main()
