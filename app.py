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
        
    def findAngle(self, img, p1, p2, p3, draw=True):   
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



    print('1: Toggle code view')
    print('2: Toggle grid')
    print('3: Backspace')
    print('4: Clear code')

    while cap.isOpened():
        ret, flipped_frame = cap.read()

        if ret:    
            frame = cv2.flip(flipped_frame, 1)
            annotated_frame = detector.findPose(frame, True)

            h, w, c = frame.shape
            # w = 640

            if show_grid_lines: 
                cv2.line(frame, (200, 0), (200, h), (111,111,111), 2)
                cv2.line(frame, (400, 0), (400, h), (111,111,111), 2)
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
                elbow_r = detector.findAngle(frame, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
                elbow_l = detector.findAngle(frame, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

                # Arms horizontal and elbows straight
                elbows_straight = elbow_l > 145 and elbow_r > 145
                left_arm_horizonal = abs(lmList[LEFT_SHOULDER][2] - lmList[LEFT_WRIST][2]) < 50
                right_arm_horizonal = abs(lmList[RIGHT_SHOULDER][2] - lmList[RIGHT_WRIST][2]) < 50
                if elbows_straight and left_arm_horizonal and right_arm_horizonal:
                    if last_command == '.':
                        same_command_count += 1
                    else:
                        last_command = '.'
                        code += last_command
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY:
                        cv2.putText(frame, '.', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)

                # Double-up, not included in original spec
                elif lmList[LEFT_WRIST][2] < lmList[NOSE][2] and lmList[RIGHT_WRIST][2] < lmList[NOSE][2]: 
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
                        cv2.putText(frame, '++', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)                                
               
                # Hands up!
                elif lmList[LEFT_WRIST][2] < lmList[NOSE][2] or lmList[RIGHT_WRIST][2] < lmList[NOSE][2]: 
                    if last_command != '++':  # Do not unintentional trigger single +, if not lowering both arms exacly at the same time
                        if last_command == '+':
                            same_command_count += 1
                        else:
                            last_command = '+'
                            if code.endswith('+++++'):
                                code += ' ' + last_command
                            else:
                                code += last_command
                            same_command_count = 0
                        if same_command_count > COMMAND_DELAY:
                            cv2.putText(frame, '+', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)                                
                
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
                        cv2.putText(frame, '-', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                
                # Body to the left
                elif lmList[LEFT_SHOULDER][1] < 200 and lmList[RIGHT_SHOULDER][1] < 200:
                    if last_command == '<' or last_command == '[':
                        same_command_count += 1
                    else:
                        last_command = '<'
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY and same_command_count < 30:
                        cv2.putText(frame, '<', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                    if same_command_count > 30 and last_command != '[':
                        last_command = '['
                        code += last_command
                        cv2.putText(frame, '[', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                
                # Body to the right
                elif lmList[LEFT_SHOULDER][1] > 400 and lmList[RIGHT_SHOULDER][1] > 400:
                    if last_command == '>' or last_command == ']':
                        same_command_count += 1
                    else:
                        last_command = '>'
                        same_command_count = 0
                    if same_command_count > COMMAND_DELAY and same_command_count < 30:
                        cv2.putText(frame, '>', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                    if same_command_count > 30 and last_command != ']':
                        last_command = ']'
                        code += last_command
                        cv2.putText(frame, ']', (200, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                else:
                    if last_command in ['<','>']:
                        code += last_command
                    same_command_count = 0
                    last_command = ''
                #elif True: # Clap, and no second clap for 1 second
                #    pass
                #elif True: # Clap twice
                #    pass

            cv2.namedWindow('BodyFuck', cv2.WINDOW_NORMAL)
            cv2.imshow('BodyFuck', annotated_frame)

        if cv2.waitKey(1) == 27:  # 27 == ESC key
            break
        elif cv2.waitKey(1) == ord('1') and last_keypress != '1':
            last_keypress = '1'
            show_code_lines = not show_code_lines
        elif cv2.waitKey(1) == ord('2') and last_keypress != '2':
            last_keypress = '2'
            show_grid_lines = not show_grid_lines
        elif cv2.waitKey(1) == ord('3') and last_keypress != '3':
            last_keypress = '3'
            code = code[:-1]
        elif cv2.waitKey(1) == ord('4') and last_keypress != '4':
            last_keypress = '4'
            code = ''
        else:
            last_keypress = '0'

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

# fiks implementering av hopp. Albue over munn? og håndledd albue, med minst halve underarms lengde?
# fiks implementering av print (.) også
# fiks visning av enkelttegn, midt på, eller i hjørne? Alpha? Rounded corners?
# implementer klapp-deteksjon
# implementer brainfuck-interpreter, og vis resultatet
# tilpass hvilke pose features som vises på video streamen
# fiks fin bakgrunn på current symbol