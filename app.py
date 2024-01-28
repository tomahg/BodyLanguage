import cv2
import mediapipe as mp
import math
from mediapipe.python.solutions.pose import PoseLandmark

class poseDetector() :    
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        
        self.mode = mode 
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

    def process (self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)        

        # Hide face landmarks
        excluded_landmarks = [
            PoseLandmark.LEFT_EYE, 
            PoseLandmark.RIGHT_EYE, 
            PoseLandmark.LEFT_EYE_INNER, 
            PoseLandmark.RIGHT_EYE_INNER, 
            PoseLandmark.LEFT_EAR,
            PoseLandmark.RIGHT_EAR,
            PoseLandmark.LEFT_EYE_OUTER,
            PoseLandmark.RIGHT_EYE_OUTER,
            PoseLandmark.NOSE,
            PoseLandmark.MOUTH_LEFT,
            PoseLandmark.MOUTH_RIGHT ]

        for landmark in excluded_landmarks:
            self.results.pose_landmarks.landmark[landmark].visibility = 0

        # Draw body landmarks
        # Default style = custom_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def find_pixel_positions(self, img):
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Determining the pixel position of the landmarks
                h, w, c = img.shape
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

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)

    FONT_SIZE = 10
    FONT_WEIGHT = 10
    MIN_CHARS_PER_LINE = 15
    MAX_LINES_OF_CODE = 4
    HORIZONTAL_MARGIN = 10

    COMMAND_DELAY = 0

    show_code_lines = True
    show_grid_lines = False

    last_command = ''
    same_command_count = 0
    last_keypress = ''
    code = ''

    clap_count = 0
    clap_stage = ''
    clap_closing_timeframe = 0
    clap_display_for_frames = 0
    clap_print1 = 0
    clap_print2 = 0

    print_lock = 0
    pause = False


    # Menu
    print('1: Toggle code view')
    print('2: Toggle grid')
    print('3: Backspace')
    print('4: Clear code')
    print('5. Pause')

    while cap.isOpened():
        ready, flipped_frame = cap.read()

        if ready:    
            frame = cv2.flip(flipped_frame, 1)
            annotated_frame = detector.process(frame)
            landmarks = detector.find_pixel_positions(frame)

            h, w, c = frame.shape
            # w = 640

            if show_grid_lines: 
                cv2.line(frame, (200, 0), (200, h), (111,111,111), 2)
                cv2.line(frame, (440, 0), (440, h), (111,111,111), 2)
                cv2.line(frame, (0, 440), (w, 440), (111,111,111), 2)

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
                        lines_of_code.append(code_left_to_print[:char_count]) 
                        code_left_to_print = code_left_to_print[char_count:]
                    else:
                        lines_of_code.append(code_left_to_print)
                        code_left_to_print = ''

                for i, line_of_code in enumerate(lines_of_code[-MAX_LINES_OF_CODE:]):
                    cv2.rectangle(frame, (0, i * 36), (w, 36 + i * 36), (0,0,0), -1)
                    cv2.putText(frame, line_of_code.strip(), (int(HORIZONTAL_MARGIN / 2), i * 36 + 28), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2) 

            if len(landmarks) and not pause:
                elbow_r = detector.find_angle(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST)
                elbow_l = detector.find_angle(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST)
                upper_arm_l = detector.find_length(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW)
                upper_arm_r = detector.find_length(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW)
                half_upper_arm = int((upper_arm_l + upper_arm_r) / 4)
                upper_arm = int((upper_arm_l + upper_arm_r) / 2)

                # Elbows straight
                elbows_straight = elbow_l > 130 and elbow_r > 130
                # Arms horizontal, less then half an upper arm off
                left_arm_horizonal = abs(landmarks[PoseLandmark.LEFT_SHOULDER][2] - landmarks[PoseLandmark.LEFT_WRIST][2]) < half_upper_arm
                right_arm_horizonal = abs(landmarks[PoseLandmark.RIGHT_SHOULDER][2] - landmarks[PoseLandmark.RIGHT_WRIST][2]) < half_upper_arm
                if elbows_straight and left_arm_horizonal and right_arm_horizonal:
                    if last_command == '.':
                        same_command_count += 1
                    elif print_lock == 0: # Avoid triggering double .
                        print_lock = 1
                        last_command = '.'
                        code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        cv2.putText(frame, '.', (290, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)

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
                        if landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2]: 
                            cv2.putText(frame, '+', (140, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT) 
                        if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2]:
                            cv2.putText(frame, '+', (380, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
            
                # Duck, shoulders below threshold
                elif landmarks[PoseLandmark.LEFT_SHOULDER][2] > 450 and landmarks[PoseLandmark.RIGHT_SHOULDER][2] > 450: 
                    if last_command == '-':
                        same_command_count += 1
                    else:
                        last_command = '-'
                        if code.endswith('----'):
                            code += ' ' + last_command
                        else:
                            code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        cv2.putText(frame, '-', (260, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                
                # Body to the left
                elif landmarks[PoseLandmark.LEFT_SHOULDER][1] < 200 and landmarks[PoseLandmark.RIGHT_SHOULDER][1] < 200:
                    if last_command == '<' or last_command == '[':
                        same_command_count += 1
                    else:
                        last_command = '<'
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY and same_command_count < 30:
                        cv2.putText(frame, '<', (260, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                    if same_command_count > 30:
                        if last_command != '[':
                            last_command = '['
                            code += last_command
                        cv2.putText(frame, '[', (280, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                
                # Body to the right
                elif landmarks[PoseLandmark.LEFT_SHOULDER][1] > 440 and landmarks[PoseLandmark.RIGHT_SHOULDER][1] > 440:
                    if last_command == '>' or last_command == ']':
                        same_command_count += 1
                    else:
                        last_command = '>'
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY and same_command_count < 30:
                        cv2.putText(frame, '>', (260, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                    if same_command_count > 30:
                        if last_command != ']':
                            last_command = ']'
                            code += last_command
                        cv2.putText(frame, ']', (280, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                else:
                    if last_command in ['<','>']:
                        code += last_command
                    last_command = ''
                    same_command_count = 0 

                    # Dummy command to identify default position with arms down
                    # Clapping should be performed in this position. buy with wrists higher then elbows
                    shoulder_r = detector.find_angle(PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW)
                    shoulder_l = detector.find_angle(PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW)
                    if shoulder_r < 60 and shoulder_l < 60: # arms facing downwards
                        # Remember that right and left are mirrored
                        if landmarks[PoseLandmark.LEFT_SHOULDER][1] < 500 and landmarks[PoseLandmark.RIGHT_SHOULDER][1] < 500: # not too far right
                            if landmarks[PoseLandmark.LEFT_SHOULDER][1] > 140 and landmarks[PoseLandmark.RIGHT_SHOULDER][1] > 140: # not too far left
                                last_command = 'default'
                                print_lock = 0 # Must return to default between each print command
                                
                    if clap_print1 == 1 or clap_print2 == 1:
                        if clap_display_for_frames > 0:
                            clap_display_for_frames -= 1
                            if clap_count >= 2:
                                cv2.putText(frame, 'Clap! Clap!', (120, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), FONT_WEIGHT)
                            elif clap_count == 1:
                                cv2.putText(frame, 'Clap!', (230, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), FONT_WEIGHT)
                        else:
                            if clap_count == 1:
                                print('Executing code...')
                                print(code)
                            clap_print1 = 0
                            clap_print2 = 0
                            clap_stage = ''
                            clap_count = 0

                    if last_command == 'default' or last_command == '':
                        if landmarks[PoseLandmark.LEFT_INDEX][1] > landmarks[PoseLandmark.LEFT_SHOULDER][1] and landmarks[PoseLandmark.RIGHT_INDEX][1] < landmarks[PoseLandmark.RIGHT_SHOULDER][1]:
                            if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                if landmarks[PoseLandmark.LEFT_SHOULDER][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_SHOULDER][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                    clap_stage = 'wide' 
                                    clap_closing_timeframe = 5    
                        if clap_stage == 'wide' and clap_closing_timeframe > 0 and abs(landmarks[PoseLandmark.LEFT_INDEX][1] - landmarks[PoseLandmark.RIGHT_INDEX][1]) < int(half_upper_arm):
                            if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                clap_stage = 'clap'
                                clap_count += 1
                                print('clap', clap_count)
                    if clap_count >= 2:
                        if clap_print2 == 0:
                            clap_display_for_frames = 10
                            clap_print2 = 1
                        code = ''
                        print('Clearing code')
                    elif clap_count == 1:
                        if clap_print1 == 0:
                            clap_display_for_frames = 10
                            clap_print1 = 1
                    elif clap_stage == 'wide' and clap_closing_timeframe > 0:
                        clap_closing_timeframe -= 1

            cv2.namedWindow('BodyFuck', cv2.WINDOW_NORMAL)
            cv2.imshow('BodyFuck', annotated_frame)

        if cv2.waitKey(1) == 27:  # 27 == ESC key
            break
        elif cv2.waitKey(1) == ord('1') and last_keypress != '1': #Toggle code view
            last_keypress = '1'
            show_code_lines = not show_code_lines
        elif cv2.waitKey(1) == ord('2') and last_keypress != '2': #Toggle grid
            last_keypress = '2'
            show_grid_lines = not show_grid_lines
        elif cv2.waitKey(1) == ord('3') and last_keypress != '3': #Backspace
            last_keypress = '3'
            code = code[:-1]
        elif cv2.waitKey(1) == ord('4') and last_keypress != '4': #Clear code
            last_keypress = '4'
            code = ''
        elif cv2.waitKey(1) == ord('5') and last_keypress != '5': #Pause
            last_keypress = '5'
            pause = not pause
            if pause:
                print('Paused')
            else:
                print('Resumed')
        else:
            last_keypress = '0'

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

# fiks visning av enkelttegn, midt på? Alpha? Rounded corners?
# implementer brainfuck-interpreter, og vis resultatet. gjerne tegn for tegn, men highlighting
# trykk tast/museknapp for å starte å ta imot kommandoer. 
# rydd opp?