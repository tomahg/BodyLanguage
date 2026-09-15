from Interpreter import Visualnterpreter
from DrawUtils import SpeechBubble
import cv2
import datetime
import json
import math
import mediapipe as mp
import os
import time
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
FACEPALM_HEIGHT_FACTOR = 2.0

# --- Timings ---------------------------------------------------------------
# Everything below is seconds. Used to be frames.

# Grace period before the overlay for a newly recognised pose is drawn. 
# 0.0 draws it immediately. Add a higher number for a delay.
COMMAND_OVERLAY_DELAY_SECONDS = 0.0

# How long to stay at the edge of the frame before < becomes [ (and > becomes ]).
BRACKET_HOLD_SECONDS = 1.0

# Recognising a clap: once the arms are spread wide, the hands have this long
# to meet, or the gesture is discarded.
CLAP_HANDS_TOGETHER_SECONDS = 1.25

# After a clap has been recognised: how long the Clap! banner stays up.
# The single-clap banner doubles as the window in which a second clap can still
# arrive and turn it into a double clap, so the code is not run until it runs
# out. The double-clap banner is display only, the stop/clear has already
# happened by then.
CLAP_SINGLE_DISPLAY_SECONDS = 1.0
CLAP_DOUBLE_DISPLAY_SECONDS = 0.5

# Interpreter pacing: shortest time between two executed brainfuck commands.
# Inside a [ ] loop it runs faster, so loops do not take forever.
INTERPRETER_STEP_SECONDS = 0.15
INTERPRETER_LOOP_STEP_SECONDS = 0.05

# --- Pose model ------------------------------------------------------------
# The landmarker runs on every single frame, so this is the biggest lever there
# is on the frame rate, and the frame rate is what decides how smooth the whole
# thing looks. 'lite' is fastest and shakiest, 'heavy' steadiest but far too
# slow to hold a frame rate, 'full' sits between them and is the one measured
# to keep up: about 34ms a frame against 142ms for heavy.
POSE_MODEL = 'full'  # 'lite', 'full' or 'heavy'

# Fast stepping in the paused debugger: after pointing, the forearm is cranked
# around the elbow to keep stepping the way the hand points.

# Degrees of cranking that earn a single step. 45 makes a full turn eight steps,
# so the faster the crank turns, the faster the debugger steps.
SPIN_DEGREES_PER_STEP = 45.0

# How far the crank has to turn, the same way, before it steps anything. Half a
# turn is more than a forearm travels between pointing and hanging down, so
# single stepping over and over is never mistaken for a crank.
SPIN_MIN_DEGREES_TO_START = 180.0

# Which way the crank turns: 1 lets each hand roll an imaginary wheel the way it
# points, so the right arm turns clockwise and the left arm counter-clockwise.
# Flip to -1 to turn both arms the other way.
SPIN_TURN_SIGN = 1

# Turning slower than this is an arm moving about, not a crank.
SPIN_MIN_DEGREES_PER_SECOND = 120.0

# No single frame can be worth more turning than this. Anything larger is the
# pose detector jumping to a new guess, not an arm that moved.
SPIN_MAX_DEGREES_PER_FRAME = 150.0

# How long the crank may stall before the spin is forgotten, and the hand has to
# point once more to start a new one.
SPIN_TIMEOUT_SECONDS = 0.4


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
COMPETITION_WORD = 'NDC'
CAMERA_INDEX = 1

_MODEL_FILES = {
    'lite': 'pose_landmarker_lite.task',
    'full': 'pose_landmarker_full.task',
    'heavy': 'pose_landmarker.task',
}
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotator', _MODEL_FILES[POSE_MODEL])

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
            # VIDEO rather than IMAGE: the landmarker tracks the pose from one
            # frame to the next, so the expensive detector stage only runs
            # again when tracking is lost. IMAGE mode redetects from scratch
            # every single frame, which is most of what a frame costs.
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)
        self._connections = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
        self.detection_result = None
        # VIDEO mode wants timestamps that never stand still or go backwards
        self.last_timestamp_ms = -1

    def process(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(imgRGB))
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        self.detection_result = self.pose.detect_for_video(mp_image, timestamp_ms)

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

def facepalm_checks(landmarks, index_landmark):
    # Index finger horizontally between the outer eyes, above nose, not too far above head
    li_x = landmarks[index_landmark][1]
    li_y = landmarks[index_landmark][2]
    left_eye_x = landmarks[PoseLandmark.LEFT_EYE_OUTER][1]
    right_eye_x = landmarks[PoseLandmark.RIGHT_EYE_OUTER][1]
    nose_y = landmarks[PoseLandmark.NOSE][2]
    face_width = abs(left_eye_x - right_eye_x)
    nose_top = nose_y - int(FACEPALM_HEIGHT_FACTOR * face_width)

    return [
        (f'Finger left of L eye  ({li_x} < {left_eye_x})', li_x < left_eye_x),
        (f'Finger right of R eye ({li_x} > {right_eye_x})', li_x > right_eye_x),
        (f'Finger above nose     ({li_y} < {nose_y})', li_y < nose_y),
        (f'Finger not too high   ({li_y} > {nose_top})', li_y > nose_top),
    ]

def is_facepalm(landmarks, index_landmark):
    return all(ok for _, ok in facepalm_checks(landmarks, index_landmark))

def draw_facepalm_overlay(img, landmarks):
    # Diagnostic overlay for the facepalm
    hands = [
        ('Right hand', PoseLandmark.LEFT_INDEX),
        ('Left hand', PoseLandmark.RIGHT_INDEX),
    ]

    x, y, box_w = 10, 10, 360
    line_h = 24
    total_lines = 1 + sum(1 + len(facepalm_checks(landmarks, lm)) for _, lm in hands)
    box_h = line_h * total_lines + 12
    draw_white_apha_box(img, x, y, box_h, box_w)

    line = 0
    cv2.putText(img, 'Facepalm requirements', (x + 8, y + 20 + line_h * line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    line += 1

    for hand_label, index_landmark in hands:
        checks = facepalm_checks(landmarks, index_landmark)
        all_ok = all(ok for _, ok in checks)
        subtitle_color = (0, 150, 0) if all_ok else (0, 0, 0)
        cv2.putText(img, hand_label + ':', (x + 8, y + 20 + line_h * line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, subtitle_color, 1)
        line += 1
        for label, ok in checks:
            color = (0, 150, 0) if ok else (0, 0, 255)
            mark = 'OK ' if ok else 'X  '
            cv2.putText(img, mark + label, (x + 8, y + 20 + line_h * line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            line += 1

def forearm_landmarks(direction):
    # Remember that left and right are mirrored: direction 1 is the arm that
    # points to the right of the screen, direction -1 the one pointing left
    if direction > 0:
        return PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST
    return PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST

def is_pointing(landmarks, direction):
    # The hand has to be past the shoulder, on the side it is pointing to
    shoulder, _, wrist = forearm_landmarks(direction)
    return (landmarks[wrist][1] - landmarks[shoulder][1]) * direction > 0

def forearm_angle(landmarks, direction):
    # Angle of the forearm around the elbow. Y grows downwards, so the angle
    # grows clockwise, the way it looks on screen
    _, elbow, wrist = forearm_landmarks(direction)
    return math.degrees(math.atan2(landmarks[wrist][2] - landmarks[elbow][2],
                                   landmarks[wrist][1] - landmarks[elbow][1]))

def angle_difference(previous, current):
    # Signed degrees from one angle to the next, the short way around
    return (current - previous + 180.0) % 360.0 - 180.0

def delete_last_command(code):
    # The spaces that break up more than five + or - are formatting,
    # not commands, so a single delete takes the space along with the command
    # in front of it. Otherwise deleting reads as if nothing happened.
    return code[:-1].rstrip(' ')

class ForearmSpin:
    """Keeps the paused debugger stepping while a forearm is cranked around the elbow.

    Only ever armed by the pointing gesture, so a crank has to start out from a
    hand pointing the way it is about to turn. The turning has to follow the
    pointing direction too, as if the hand rolled a wheel that way, which makes
    the right arm turn clockwise and the left arm counter-clockwise. Every
    SPIN_DEGREES_PER_STEP of turning is worth one step, so cranking twice as
    fast steps twice as fast.
    """

    def __init__(self):
        self.direction = 0
        self.forget()

    def forget(self):
        self.last_angle = None
        self.last_time = 0.0
        self.forget_turning()

    def forget_turning(self):
        self.wound_up = 0.0
        self.unspent = 0.0
        self.turning_until = 0.0
        self.spinning_until = 0.0

    def arm(self, direction):
        if direction != self.direction:
            self.direction = direction
            self.forget()

    def disarm(self):
        self.arm(0)

    def is_spinning(self, now):
        return self.direction != 0 and now < self.spinning_until

    def update(self, landmarks, upper_arm, now):
        """Feed the current pose, and get the steps the crank has earned since the last frame."""
        if self.direction == 0:
            return 0

        # The elbow stays out to the side while the forearm swings around it,
        # so an arm that is lowered is no longer cranking
        shoulder, elbow, _ = forearm_landmarks(self.direction)
        elbow_out = (landmarks[elbow][1] - landmarks[shoulder][1]) * self.direction > 0
        elbow_level = abs(landmarks[elbow][2] - landmarks[shoulder][2]) < upper_arm
        if not (elbow_out and elbow_level):
            self.disarm()
            return 0

        angle = forearm_angle(landmarks, self.direction)
        elapsed = now - self.last_time
        if self.last_angle == None or elapsed <= 0:
            self.last_angle = angle
            self.last_time = now
            return 0

        turned = angle_difference(self.last_angle, angle) * self.direction * SPIN_TURN_SIGN
        self.last_angle = angle
        self.last_time = now

        # More than this in a single frame is the pose detector jumping to a new
        # guess rather than an arm that moved
        if abs(turned) > SPIN_MAX_DEGREES_PER_FRAME:
            return 0

        # Turning the way the hand pointed winds the crank up, turning back
        # unwinds it again. Swinging the forearm down to point once more unwinds
        # every bit of what swinging it up wound up, which is what keeps single
        # stepping over and over from turning into a crank
        self.wound_up = max(0.0, self.wound_up + turned)

        # Slower than this is an arm moving about rather than a crank. Hold on to
        # the wind-up for a moment, in case the crank is only passing through a
        # slow patch, and forget it once the turning has really stopped
        if turned / elapsed < SPIN_MIN_DEGREES_PER_SECOND:
            if now >= self.turning_until:
                self.forget_turning()
            return 0
        self.turning_until = now + SPIN_TIMEOUT_SECONDS

        # A crank has to get properly going before it steps anything
        if self.wound_up < SPIN_MIN_DEGREES_TO_START:
            return 0

        self.spinning_until = now + SPIN_TIMEOUT_SECONDS
        self.unspent += turned
        steps = int(self.unspent / SPIN_DEGREES_PER_STEP)
        self.unspent -= steps * SPIN_DEGREES_PER_STEP
        return steps

def main():
    global CAMERA_INDEX
    global SHOW_GRID_LINES

    detector = PoseDetector()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}. Releasing and retrying...")
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print(f"Still could not open camera {CAMERA_INDEX}. Giving up.")
            cap.release()
            return

    show_code_lines = True
    show_facepalm_overlay = False
    SHOW_GRID_LINES = False

    last_command = ''
    command_started_at = time.perf_counter()
    code = ''
    lines_of_code = []

    clap_count = 0
    clap_stage = ''
    clap_hands_together_deadline = 0.0
    clap_display_until = 0.0
    clap_print1 = 0
    clap_print2 = 0

    print_lock = 0
    facepalm_lock = 0
    pause = False
    execute_code = False
    code_output = ''
    interpreter_finished_debug_and_print = False
    interpreter_paused = False
    interpreter_stopped = False
    interpreter_error = False
    interpreter_error_line = 0
    interpreter_error_char = 0
    # Steps the paused debugger owes: positive forwards, negative backwards
    pending_steps = 0
    forearm_spin = ForearmSpin()
    interpreter = Visualnterpreter()
    interpreter.step_interval_seconds = INTERPRETER_STEP_SECONDS
    interpreter.loop_step_interval_seconds = INTERPRETER_LOOP_STEP_SECONDS
    speech_bubble = SpeechBubble()

    competition_start_time = None
    competition_end_time = None
    total_time = ''
    competition_started = False
    competition_word_printed = False
    fullscreen = False

    # Menu
    print(' ')
    print('        c: Toggle code view')
    print('        f: Toggle facepalm overlay')
    print('        g: Toggle grid')
    print('        p: Pause')
    print('backspace: Delete single character')
    print('   delete: Clear code')
    print('      F11: Toggle fullscreen')
    print(' ')

    try:
        while cap.isOpened():
            ready, flipped_frame = cap.read()

            if ready:    
                now = time.perf_counter()
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
                    # The end of the program is the end. Stepping forwards stops at
                    # the last character, so a crank that keeps turning after the
                    # program ran out does not pile up steps to nowhere. Travelling
                    # back in time is all that is left from here.
                    if interpreter_finished_debug_and_print and pending_steps > 0:
                        pending_steps = 0
                    if not interpreter_paused and not interpreter_stopped and not interpreter_finished_debug_and_print and not pause or (interpreter_paused and pending_steps != 0):
                        # A crank can be worth more than a single step per frame
                        if interpreter_paused:
                            steps = abs(pending_steps)
                            forwards = pending_steps > 0
                        else:
                            steps = 1
                            forwards = True
                        pending_steps = 0

                        complete_outout = None
                        for _ in range(steps):
                            o = ''
                            if not interpreter_paused:
                                finished, remember, c, l, o = interpreter.step()
                            elif forwards:
                                finished, remember, c, l, o = interpreter.step(single_step=True)
                            else:
                                interpreter_finished_debug_and_print = False
                                finished, remember, c, l, complete_outout = interpreter.step_back()

                            if o:
                                code_output += o

                            if complete_outout != None:
                                code_output = complete_outout

                            if remember:
                                interpreter.history_append(code_output)

                            if finished:
                                break

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

                    code_changed = lines_of_code != interpreter.code
                    interpreter.input_code(lines_of_code)

                    if code_changed:
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

                    # Diagnostic overlay: show which facepalm requirements are (un)met.
                    if show_facepalm_overlay:
                        draw_facepalm_overlay(frame, landmarks)

                    # Elbow positions
                    elbow_left_straight = elbow_l > 130
                    elbow_right_straight = elbow_r > 130
                    elbows_straight = elbow_left_straight and elbow_right_straight
                    # Arms horizontal, less then half an upper arm off
                    left_arm_horizonal = abs(landmarks[PoseLandmark.LEFT_SHOULDER][2] - landmarks[PoseLandmark.LEFT_WRIST][2]) < half_upper_arm
                    right_arm_horizonal = abs(landmarks[PoseLandmark.RIGHT_SHOULDER][2] - landmarks[PoseLandmark.RIGHT_WRIST][2]) < half_upper_arm

                    # Fast stepping: cranking the forearm around the elbow, having
                    # first pointed the way, keeps the paused debugger stepping
                    if execute_code and interpreter_paused:
                        pending_steps += forearm_spin.update(landmarks, upper_arm, now) * forearm_spin.direction
                    else:
                        forearm_spin.disarm()

                    # Cranking is a debugger gesture, not code input, so while it
                    # lasts the rest of the command chain is deliberately skipped:
                    # an arm swinging past the raise and duck poses neither types
                    # anything nor draws a command on screen.
                    if forearm_spin.is_spinning(now):
                        pass

                    # Arms out, printing
                    elif elbows_straight and left_arm_horizonal and right_arm_horizonal:
                        if last_command != '.' and print_lock == 0: # Avoid triggering double .
                            print_lock = 1
                            last_command = '.'
                            code += last_command
                            command_started_at = now
                        if now - command_started_at >= COMMAND_OVERLAY_DELAY_SECONDS:
                            draw_white_apha_box(frame, 260, 95, 110, 120)
                            # Print . a litle higher than other commands
                            cv2.putText(frame, '.', (280+15, 200-30), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)

                    # Stepping debugger forward/back, and arming the spin
                    elif interpreter_paused and ((elbow_left_straight and left_arm_horizonal) or (elbow_right_straight and right_arm_horizonal)):
                        # Remberer that left and right are mirrored
                        if elbow_left_straight and left_arm_horizonal and not (elbow_right_straight and right_arm_horizonal):
                            if is_pointing(landmarks, 1):
                                forearm_spin.arm(1)
                            if last_command == 'default' or last_command == '':
                                last_command = '-->'
                                command_started_at = now
                                pending_steps = 1
                        elif elbow_right_straight and right_arm_horizonal and not (elbow_left_straight and left_arm_horizonal):
                            if is_pointing(landmarks, -1):
                                forearm_spin.arm(-1)
                            if last_command == 'default' or last_command == '':
                                last_command = '<--'
                                command_started_at = now
                                pending_steps = -1

                    # Double-up, not included in original spec
                    elif landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm: 
                        if last_command == '+': # Upgrading directly from + to ++, should yield a total of ++ not +++
                            last_command = '++'
                            if code.endswith('+++++'):
                                code += ' +'
                            else:
                                code += '+'
                        elif last_command != '++':
                            last_command = '++'
                            if code.endswith('+++++'):
                                code += ' ++'
                            elif code.endswith('++++'):
                                code += '+ +'
                            else:
                                code += last_command
                            command_started_at = now
                        if now - command_started_at >= COMMAND_OVERLAY_DELAY_SECONDS:
                            draw_white_apha_box(frame, 145, 95, 110, 355)
                            cv2.putText(frame, '+', (140, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
                            cv2.putText(frame, '+', (380, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)                                
               
                    # Hands up!
                    elif landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm or landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm : 
                        # '++' keeps its timer running, so a single + is not unintentionally
                        # triggered when not lowering both arms exacly at the same time
                        if last_command != '+' and last_command != '++':
                            last_command = '+'
                            if code.endswith('+++++'):
                                code += ' ' + last_command
                            else:
                                code += last_command
                            command_started_at = now

                        if now - command_started_at >= COMMAND_OVERLAY_DELAY_SECONDS:
                            if landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm: 
                                draw_white_apha_box(frame, 145, 95, 110, 120)
                                cv2.putText(frame, '+', (140, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT) 
                            if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.NOSE][2] - upper_arm:
                                draw_white_apha_box(frame, 385, 95, 110, 120)
                                cv2.putText(frame, '+', (380, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
            
                    # Duck, shoulders below threshold
                    elif landmarks[PoseLandmark.LEFT_SHOULDER][2] > THRESHOLD_DUCK_Y and landmarks[PoseLandmark.RIGHT_SHOULDER][2] > THRESHOLD_DUCK_Y: 
                        if last_command != '-':
                            last_command = '-'
                            if code.endswith('-----'):
                                code += ' ' + last_command
                            else:
                                code += last_command
                            command_started_at = now
                        if now - command_started_at >= COMMAND_OVERLAY_DELAY_SECONDS:
                            draw_white_apha_box(frame, 260, 95, 110, 120)
                            cv2.putText(frame, '-', (280-20, 200-5), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)   
                
                    # Body to the left
                    elif landmarks[PoseLandmark.LEFT_SHOULDER][1] < THRESHOLD_LEFT_X and landmarks[PoseLandmark.RIGHT_SHOULDER][1] < THRESHOLD_LEFT_X:
                        if last_command != '<' and last_command != '[':
                            last_command = '<'
                            command_started_at = now
                        held_seconds = now - command_started_at
                        if held_seconds >= COMMAND_OVERLAY_DELAY_SECONDS and held_seconds < BRACKET_HOLD_SECONDS:
                            draw_white_apha_box(frame, 260, 95, 110, 120)
                            cv2.putText(frame, '<', (260+5, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
                        if held_seconds > BRACKET_HOLD_SECONDS:
                            if last_command != '[':
                                last_command = '['
                                code += last_command
                            draw_white_apha_box(frame, 260, 95, 110, 120)
                            # [ is strangely large, print it a little smaller, and further up, than other commands
                            cv2.putText(frame, '[', (280+20, 200-25), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE - 5, (0,0,255), FONT_WEIGHT)   
                
                    # Body to the right
                    elif landmarks[PoseLandmark.LEFT_SHOULDER][1] > THRESHOLD_RIGHT_X and landmarks[PoseLandmark.RIGHT_SHOULDER][1] > THRESHOLD_RIGHT_X:
                        if last_command != '>' and last_command != ']':
                            last_command = '>'
                            command_started_at = now
                        held_seconds = now - command_started_at
                        if held_seconds >= COMMAND_OVERLAY_DELAY_SECONDS and held_seconds < BRACKET_HOLD_SECONDS:
                            draw_white_apha_box(frame, 260, 95, 110, 120)
                            cv2.putText(frame, '>', (260+5, 200), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,255), FONT_WEIGHT)
                        if held_seconds > BRACKET_HOLD_SECONDS:
                            if last_command != ']':
                                last_command = ']'
                                code += last_command
                            draw_white_apha_box(frame, 260, 95, 110, 120)
                            # ] is strangely large, print it a little smaller, and further up, than other commands
                            cv2.putText(frame, ']', (280+20, 200-25), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE - 5, (0,0,255), FONT_WEIGHT)   

                    # Facepalm
                    # Index finger horizontally between the outer eyes, above eyes, not too far above head
                    elif is_facepalm(landmarks, PoseLandmark.LEFT_INDEX) \
                            or is_facepalm(landmarks, PoseLandmark.RIGHT_INDEX):
                        if last_command != '⌫':
                            if facepalm_lock == 0:
                                facepalm_lock = 1
                                command_started_at = now
                                code = delete_last_command(code)
                            last_command = '⌫'
                        bubble_x = landmarks[PoseLandmark.MOUTH_RIGHT][1]
                        bubble_y = landmarks[PoseLandmark.MOUTH_RIGHT][2]
                        speech_bubble.draw(frame, bubble_x, bubble_y, half_upper_arm)
                    else:
                        if last_command in ['<','>']:
                            code += last_command
                        last_command = ''
                        command_started_at = now

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
                                    facepalm_lock = 0
                                
                        if clap_print1 == 1 or clap_print2 == 1:
                            if now < clap_display_until:
                                if clap_count >= 2:
                                    draw_white_apha_box(frame, 120, 95, 110, 400)
                                    cv2.putText(frame, 'Clap! Clap!', (150, 180), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), FONT_WEIGHT)
                                elif clap_count == 1:
                                    draw_white_apha_box(frame, 200-10, 95, 110, 220)
                                    cv2.putText(frame, 'Clap!', (225, 180), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), FONT_WEIGHT)
                            else:
                                if clap_count == 1:
                                    competition_end_time = datetime.datetime.now()
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
                                        clap_hands_together_deadline = now + CLAP_HANDS_TOGETHER_SECONDS
                            if clap_stage == 'wide' and now < clap_hands_together_deadline and abs(landmarks[PoseLandmark.LEFT_INDEX][1] - landmarks[PoseLandmark.RIGHT_INDEX][1]) < int(half_upper_arm):
                                if landmarks[PoseLandmark.LEFT_WRIST][2] < landmarks[PoseLandmark.LEFT_ELBOW][2] and landmarks[PoseLandmark.RIGHT_WRIST][2] < landmarks[PoseLandmark.RIGHT_ELBOW][2]:
                                    clap_stage = 'clap'
                                    clap_count += 1
                        if clap_count >= 2:
                            if clap_print2 == 0:
                                clap_display_until = now + CLAP_DOUBLE_DISPLAY_SECONDS
                                clap_print2 = 1
                                # If interpreter is running: stop it
                                # Otherwise clear code buffer
                                if execute_code:
                                    execute_code = False
                                    interpreter_paused = False
                                else:
                                    code = ''
                                    code_output = ''
                                    competition_end_time = None                 
                        elif clap_count == 1:
                            if clap_print1 == 0:
                                clap_display_until = now + CLAP_SINGLE_DISPLAY_SECONDS
                                clap_print1 = 1
                                # Pause / resume debugger immediately, without waiting for potential second clap
                                if execute_code and not pause and (interpreter_paused or not interpreter_finished_debug_and_print):
                                    if interpreter_paused:
                                        interpreter_paused = False
                                    else:
                                        interpreter_paused = True

                if last_command not in ['default', '']:
                    competition_end_time = None

                if COMPETITION_MODE:
                    if competition_started or competition_word_printed:
                        if competition_word_printed:
                            elapsed = total_time
                        else:
                            if competition_end_time == None:
                                elapsed = datetime.datetime.now() - competition_start_time
                            else:
                                elapsed = competition_end_time - competition_start_time

                        score_color = (0,255,0)
                        formatted_time = f"{elapsed.seconds // 60}:{elapsed.seconds % 60:02}"
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
                elif key == ord('f') or key == ord('F'): #Toggle facepalm overlay
                    show_facepalm_overlay = not show_facepalm_overlay
                elif key == ord('g') or key == ord('G'): #Toggle grid
                    SHOW_GRID_LINES = not SHOW_GRID_LINES
                elif key == 8: #Backspace
                    code = delete_last_command(code)
                    if code == '':
                        competition_end_time = None
                elif key == 3014656 or key == 2555904: #Clear code (delete key or right arrow / clicker)
                    code = ''
                    code_output = ''
                    execute_code = False
                    interpreter_paused = False
                    competition_end_time = None
                    # Make sure cells at the bottom of the screen is hidden
                    ok, (interpreter_error_line, interpreter_error_char) = interpreter.prepare_code()
                    # un-pause
                    pause = False
                elif key == ord('p') or key == ord('P'): #Pause
                    pause = not pause
                elif key == 7995392: #F11
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty("BodyFuck", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty("BodyFuck", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            if len(code) == 0:
                competition_started = False
                competition_word_printed = False
            elif competition_started == False:
                competition_started = True
                competition_word_printed = False
                competition_start_time = datetime.datetime.now()
                competition_end_time = None

            if COMPETITION_MODE:
                if code_output == COMPETITION_WORD and competition_word_printed == False: 
                    if competition_end_time == None:
                        total_time = datetime.datetime.now() - competition_start_time
                    else:
                        total_time = competition_end_time - competition_start_time
                    time_spent_seconds = int(total_time.total_seconds())

                    competition_word_printed = True
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

    finally:
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
