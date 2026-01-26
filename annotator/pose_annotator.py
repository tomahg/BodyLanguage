"""
Video Pose Annotator using MediaPipe
Processes video files and overlays pose landmarks detected by MediaPipe.

Configuration options are at the top of the file for easy modification.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import os
import subprocess
import shutil

# =============================================================================
# CONFIGURATION - Modify these settings as needed
# =============================================================================

# Debug mode - set to True to output a single frame image instead of full video
# Useful for quickly testing different color/thickness settings
DEBUG_SINGLE_FRAME = False
DEBUG_OUTPUT_IMAGE = r"C:\Dev\BodyFuck\annotator\debug_frame.png"  # Debug image output path

# Input/Output settings
INPUT_VIDEO = r"C:\Dev\BodyFuck\annotator\IMG_3498.MOV"  # Path to input video
OUTPUT_VIDEO = r"C:\Dev\BodyFuck\annotator\IMG_3498_annotated.mp4"  # Path to output video

# Landmark visualization settings
LANDMARK_COLOR = (175, 175, 175)  # BGR format - Green
LANDMARK_RADIUS = 10  # Size of landmark circles in pixels
LANDMARK_THICKNESS = -1  # -1 for filled circles, or positive int for outline

# Connection visualization settings
CONNECTION_COLOR = (100, 100, 100)  # BGR format - White
CONNECTION_THICKNESS = 5  # Line thickness in pixels

# MediaPipe Pose detection settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 2  # 0, 1, or 2 (higher = more accurate but slower)

# =============================================================================
# LANDMARK SELECTION
# Set to True to include, False to exclude each landmark
# =============================================================================

# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_VISIBILITY = {
    # Face landmarks
    0: False,   # nose
    1: False,   # left_eye_inner
    2: False,   # left_eye
    3: False,   # left_eye_outer
    4: False,   # right_eye_inner
    5: False,   # right_eye
    6: False,   # right_eye_outer
    7: False,   # left_ear
    8: False,   # right_ear
    9: False,   # mouth_left
    10: False,  # mouth_right
    
    # Upper body landmarks
    11: True,  # left_shoulder
    12: True,  # right_shoulder
    13: True,  # left_elbow
    14: True,  # right_elbow
    15: True,  # left_wrist
    16: True,  # right_wrist
    17: False,  # left_pinky
    18: False,  # right_pinky
    19: False,  # left_index
    20: False,  # right_index
    21: False,  # left_thumb
    22: False,  # right_thumb
    
    # Lower body landmarks
    23: True,  # left_hip
    24: True,  # right_hip
    25: True,  # left_knee
    26: True,  # right_knee
    27: True,  # left_ankle
    28: True,  # right_ankle
    29: True,  # left_heel
    30: True,  # right_heel
    31: True,  # left_foot_index
    32: True,  # right_foot_index
}

# Landmark names for reference
LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# =============================================================================
# CONNECTION SELECTION
# Define which connections to draw between landmarks
# Each tuple is (start_landmark_idx, end_landmark_idx)
# Set DRAW_CONNECTIONS to False to disable all connections
# =============================================================================

DRAW_CONNECTIONS = True

# You can customize which connections to draw by modifying this list
# Default connections follow the MediaPipe pose skeleton
CUSTOM_CONNECTIONS: Optional[List[Tuple[int, int]]] = None  # Set to None to use default

# If CUSTOM_CONNECTIONS is None, these default connections will be used:
DEFAULT_CONNECTIONS = [
    # Face
    #(0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
    #(0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
    #(9, 10),  # Mouth
    
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24),  # Hips
    
    # Left arm
    (11, 13), (13, 15),  # Shoulder to wrist
    #(15, 17), (15, 19), (15, 21),  # Wrist to fingers
    #(17, 19),  # Pinky to index
    
    # Right arm
    (12, 14), (14, 16),  # Shoulder to wrist
    #(16, 18), (16, 20), (16, 22),  # Wrist to fingers
    #(18, 20),  # Pinky to index
    
    # Left leg
    (23, 25), (25, 27),  # Hip to ankle
    (27, 29), (27, 31), (29, 31),  # Ankle to foot
    
    # Right leg
    (24, 26), (26, 28),  # Hip to ankle
    (28, 30), (28, 32), (30, 32),  # Ankle to foot
]

# =============================================================================
# ADVANCED: Per-landmark and per-connection customization
# Override colors/sizes for specific landmarks or connections
# =============================================================================

# Per-landmark color overrides (landmark_idx: BGR color)
# Example: {0: (0, 0, 255)} makes the nose red
LANDMARK_COLOR_OVERRIDES: dict = {}

# Per-landmark radius overrides (landmark_idx: radius)
# Example: {0: 10} makes the nose landmark larger
LANDMARK_RADIUS_OVERRIDES: dict = {}

# Per-connection color overrides ((start, end): BGR color)
# Example: {(11, 12): (0, 0, 255)} makes shoulder connection red
CONNECTION_COLOR_OVERRIDES: dict = {}

# Per-connection thickness overrides ((start, end): thickness)
CONNECTION_THICKNESS_OVERRIDES: dict = {}


# =============================================================================
# PROCESSING CODE - Generally no need to modify below this line
# =============================================================================

def get_visible_landmarks() -> List[int]:
    """Get list of landmark indices that should be drawn."""
    return [idx for idx, visible in LANDMARK_VISIBILITY.items() if visible]


def get_connections() -> List[Tuple[int, int]]:
    """Get list of connections to draw."""
    if not DRAW_CONNECTIONS:
        return []
    
    connections = CUSTOM_CONNECTIONS if CUSTOM_CONNECTIONS is not None else DEFAULT_CONNECTIONS
    
    # Filter connections to only include visible landmarks
    visible = set(get_visible_landmarks())
    return [(a, b) for a, b in connections if a in visible and b in visible]


def draw_landmarks_on_frame(
    frame,
    landmarks,
    frame_width: int,
    frame_height: int
):
    """Draw pose landmarks and connections on a frame."""
    
    if landmarks is None or len(landmarks) == 0:
        return frame
    
    # Convert normalized coordinates to pixel coordinates
    def to_pixel(landmark):
        return (
            int(landmark.x * frame_width),
            int(landmark.y * frame_height)
        )
    
    landmark_points = {
        idx: to_pixel(landmarks[idx])
        for idx in range(len(landmarks))
    }
    
    # Draw connections first (so landmarks appear on top)
    for start_idx, end_idx in get_connections():
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmark_points[start_idx]
            end_point = landmark_points[end_idx]
            
            # Get color and thickness (with possible overrides)
            conn_key = (start_idx, end_idx)
            color = CONNECTION_COLOR_OVERRIDES.get(conn_key, CONNECTION_COLOR)
            thickness = CONNECTION_THICKNESS_OVERRIDES.get(conn_key, CONNECTION_THICKNESS)
            
            cv2.line(frame, start_point, end_point, color, thickness)
    
    # Draw landmarks
    for idx in get_visible_landmarks():
        if idx < len(landmarks):
            point = landmark_points[idx]
            
            # Get color and radius (with possible overrides)
            color = LANDMARK_COLOR_OVERRIDES.get(idx, LANDMARK_COLOR)
            radius = LANDMARK_RADIUS_OVERRIDES.get(idx, LANDMARK_RADIUS)
            
            cv2.circle(frame, point, radius, color, LANDMARK_THICKNESS)
    
    return frame


def download_model_if_needed(model_path: str) -> str:
    """Download the pose landmarker model if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to: {model_path}")
    return model_path


def merge_audio_with_video(original_video: Path, processed_video: Path, output_path: Path):
    """Merge audio from original video with processed video using FFmpeg."""
    
    # Check if FFmpeg is available - try common Windows install locations too
    ffmpeg_path = shutil.which("ffmpeg")
    
    # If not in PATH, check common installation directories on Windows
    if ffmpeg_path is None:
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break
    
    if ffmpeg_path is None:
        print("\nWarning: FFmpeg not found. Output video will not have audio.")
        print("Install FFmpeg and add it to PATH to preserve audio.")
        # Just rename the processed video to output
        if processed_video != output_path:
            shutil.move(str(processed_video), str(output_path))
        return
    
    print(f"\nMerging audio from original video... (using {ffmpeg_path})")
    
    # FFmpeg command to copy video stream and audio stream
    cmd = [
        ffmpeg_path,
        "-i", str(processed_video),  # Processed video (no audio)
        "-i", str(original_video),    # Original video (with audio)
        "-c:v", "copy",               # Copy video stream without re-encoding
        "-c:a", "aac",                # Encode audio as AAC for mp4 compatibility
        "-map", "0:v:0",              # Take video from first input
        "-map", "1:a:0",              # Take audio from second input
        "-shortest",                  # Match duration to shortest stream
        "-y",                         # Overwrite output file
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # Remove temporary file
        if processed_video.exists() and processed_video != output_path:
            processed_video.unlink()
        print("Audio merged successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nWarning: FFmpeg failed to merge audio.")
        print(f"FFmpeg stderr: {e.stderr}")
        print("Output video saved without audio.")
        # Keep the processed video as output
        if processed_video != output_path:
            shutil.move(str(processed_video), str(output_path))


def process_video(input_path: str, output_path: str):
    """Process video file and add pose overlay."""
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download model if needed
    model_path = str(input_path.parent / "pose_landmarker.task")
    download_model_if_needed(model_path)
    
    # Initialize MediaPipe Pose Landmarker with new Tasks API
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    
    pose = vision.PoseLandmarker.create_from_options(options)
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {input_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    # Handle rotation metadata for iPhone videos
    # iPhone MOV files may have rotation metadata
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    print(f"Rotation metadata: {rotation}")
    
    # Debug mode: seek to middle frame and process single image
    if DEBUG_SINGLE_FRAME:
        middle_frame = total_frames // 2
        print(f"\nDebug mode: extracting frame {middle_frame} of {total_frames}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read frame {middle_frame}")
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image from frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int((middle_frame / fps) * 1000)
        
        # Process frame with MediaPipe Pose Landmarker
        results = pose.detect_for_video(mp_image, timestamp_ms)
        
        # Draw landmarks on frame
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            frame = draw_landmarks_on_frame(
                frame,
                results.pose_landmarks[0],
                frame_width,
                frame_height
            )
        else:
            print("Warning: No pose detected in this frame")
        
        # Save debug image
        debug_path = Path(DEBUG_OUTPUT_IMAGE)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), frame)
        
        cap.release()
        pose.close()
        
        print(f"Debug frame saved to: {debug_path}")
        return
    
    # Create temporary video file (without audio)
    temp_video_path = output_path.parent / f"_temp_{output_path.name}"
    
    # Create video writer
    # Using mp4v codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(temp_video_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {temp_video_path}")
    
    frame_count = 0
    
    print("\nProcessing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image from frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Calculate timestamp in milliseconds
            timestamp_ms = int((frame_count / fps) * 1000)
            
            # Process frame with MediaPipe Pose Landmarker
            results = pose.detect_for_video(mp_image, timestamp_ms)
            
            # Draw landmarks on frame
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                frame = draw_landmarks_on_frame(
                    frame,
                    results.pose_landmarks[0],  # First detected pose
                    frame_width,
                    frame_height
                )
            
            # Write frame to output
            out.write(frame)
            
            # Print progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    finally:
        cap.release()
        out.release()
        pose.close()
    
    print(f"\nVideo processing complete. Processed {frame_count} frames")
    
    # Merge audio from original video
    merge_audio_with_video(input_path, temp_video_path, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Video Pose Annotator")
    print("=" * 60)
    
    # Print configuration summary
    visible_landmarks = get_visible_landmarks()
    connections = get_connections()
    
    print(f"\nConfiguration:")
    print(f"  - Visible landmarks: {len(visible_landmarks)}/33")
    print(f"  - Connections to draw: {len(connections)}")
    print(f"  - Landmark color: BGR{LANDMARK_COLOR}")
    print(f"  - Landmark radius: {LANDMARK_RADIUS}px")
    print(f"  - Connection color: BGR{CONNECTION_COLOR}")
    print(f"  - Connection thickness: {CONNECTION_THICKNESS}px")
    print(f"  - Model complexity: {MODEL_COMPLEXITY}")
    print()
    
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
