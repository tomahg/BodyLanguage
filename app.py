import cv2
import mediapipe as mp
import math
from mediapipe.python.solutions.pose import PoseLandmark
import math

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
        
        
    def findPose (self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #finding height, width of the image printed
                h, w, c = img.shape
                #Determining the pixels of the landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList
        
    def find_angle(self, img, p1, p2, p3, draw=True):   
        #Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        #Calculate Angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
        
        #Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)
            
            cv2.circle(img, (x1, y1), 5, (0,0,255), cv2.FILLED)
            #cv2.circle(img, (x1, y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 5, (0,0,255), cv2.FILLED)
            #cv2.circle(img, (x3, y3), 15, (0,0,255), 2)
            
            #cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle
    
    def find_length(self, p1, p2):
        #Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
def get_text_width(text, font_face, font_scale, font_line_thickness):
    ((txt_w, _), _) = cv2.getTextSize(text, font_face, font_scale, font_line_thickness)
    return txt_w

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)

    # Pose landmarks
    NOSE = PoseLandmark.NOSE
    LEFT_SHOULDER = PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER
    LEFT_ELBOW = PoseLandmark.LEFT_ELBOW
    RIGHT_ELBOW = PoseLandmark.RIGHT_ELBOW
    LEFT_WRIST = PoseLandmark.LEFT_WRIST
    RIGHT_WRIST = PoseLandmark.RIGHT_WRIST
    LEFT_HIP = PoseLandmark.LEFT_HIP
    RIGHT_HIP = PoseLandmark.RIGHT_HIP

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


    # Menu
    print('1: Toggle code view')
    print('2: Toggle grid')
    print('3: Backspace')
    print('4: Clear code')

    while cap.isOpened():
        ready, flipped_frame = cap.read()

        if ready:    
            frame = cv2.flip(flipped_frame, 1)
            annotated_frame = detector.findPose(frame, True)

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

            lmList = detector.findPosition(frame, False)
            if len(lmList):
                elbow_r = detector.find_angle(frame, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
                elbow_l = detector.find_angle(frame, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
                upper_arm_l = detector.find_length(RIGHT_SHOULDER, RIGHT_ELBOW)
                upper_arm_r = detector.find_length(RIGHT_SHOULDER, RIGHT_ELBOW)
                half_upper_arm = int((upper_arm_l + upper_arm_r) / 4)
                upper_arm = int((upper_arm_l + upper_arm_r) / 2)

                # Elbows straight
                elbows_straight = elbow_l > 130 and elbow_r > 130
                # Arms horizontal, less then half an upper arm off
                left_arm_horizonal = abs(lmList[LEFT_SHOULDER][2] - lmList[LEFT_WRIST][2]) < half_upper_arm
                right_arm_horizonal = abs(lmList[RIGHT_SHOULDER][2] - lmList[RIGHT_WRIST][2]) < half_upper_arm
                if elbows_straight and left_arm_horizonal and right_arm_horizonal:
                    if last_command == '.':
                        same_command_count += 1
                    elif last_command == 'default': # Avoid triggering double .
                        last_command = '.'
                        code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        cv2.putText(frame, '.', (290, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)

                # Double-up, not included in original spec
                elif lmList[LEFT_WRIST][2] < lmList[NOSE][2] - upper_arm and lmList[RIGHT_WRIST][2] < lmList[NOSE][2] - upper_arm: 
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
                elif lmList[LEFT_WRIST][2] < lmList[NOSE][2] - upper_arm or lmList[RIGHT_WRIST][2] < lmList[NOSE][2] - upper_arm : 
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
                        if lmList[RIGHT_WRIST][2] < lmList[NOSE][2]: 
                            cv2.putText(frame, '+', (140, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT) 
                        if lmList[LEFT_WRIST][2] < lmList[NOSE][2]:
                            cv2.putText(frame, '+', (380, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
            
                # Duck, shoulders below threshold
                elif lmList[LEFT_SHOULDER][2] > 450 and lmList[RIGHT_SHOULDER][2] > 450: 
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
                elif lmList[LEFT_SHOULDER][1] < 200 and lmList[RIGHT_SHOULDER][1] < 200:
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
                elif lmList[LEFT_SHOULDER][1] > 440 and lmList[RIGHT_SHOULDER][1] > 440:
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
                    same_command_count = 0 

                    if (last_command != 'default'):
                        # Dummy command to identify default position with arms down
                        shoulder_r = detector.find_angle(frame, LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
                        shoulder_l = detector.find_angle(frame, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
                        if shoulder_r < 60 and shoulder_l < 60:
                            if lmList[LEFT_SHOULDER][1] < 440 and lmList[RIGHT_SHOULDER][1] < 440: # not too far right
                                if lmList[LEFT_SHOULDER][1] > 200 and lmList[RIGHT_SHOULDER][1] > 200: # not too far left
                                    last_command = 'default'
                                    print('default')
                        

                #elif True: # Clap, and no second clap for 1 second
                #    pass
                #elif True: # Clap twice
                #    pass

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
        else:
            last_keypress = '0'

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

# fiks visning av enkelttegn, midt på, eller i hjørne? Alpha? Rounded corners?
# implementer klapp-deteksjon
# implementer brainfuck-interpreter, og vis resultatet. gjerne tegn for tegn, men highlighting
# tilpass hvilke pose features som vises på video streamen
# trykk tast/museknapp for å starte å ta imot kommandoer. 
# rydd opp?