import cv2
import numpy as np

class SpeechBubble:
    def draw(self, image, bubble_x, bubble_y, scale):
        """
        Draws a cartoon-style speech bubble on an image.

        Parameters:
        - image: The image on which to draw the speech bubble.
        - bubble_x: The x value of bottom right corner of the bubble.
        - bubble_y: The y value of bottom right corner of the bubble.
        - scale: Some value relative to body size, for scaling the bubble.
        """
        # Bubble parameters
        text = "Whoops!"
        bubble_color = (255, 255, 255)  # White
        text_color = (0, 0, 0)  # Black
        font_scale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        radius = int(text_size[1] / 2)  # Radius for rounded corners
        bubble_size = (int(text_size[0] * 1.25), int(text_size[1] * 3)) # A tuple (width, height) representing the size of the bubble.

        # Draw the main body of the bubble with rounded corners
        bottom_right = (bubble_x, bubble_y - scale)
        top_left = (bubble_x - bubble_size[0], bubble_y - bubble_size[1] - scale)

        # Rounded rectangle
        cv2.rectangle(image, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]), bubble_color, -1)
        cv2.rectangle(image, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), bubble_color, -1)
        cv2.circle(image, (top_left[0] + radius, top_left[1] + radius), radius, bubble_color, -1)
        cv2.circle(image, (bottom_right[0] - radius, top_left[1] + radius), radius, bubble_color, -1)
        cv2.circle(image, (top_left[0] + radius, bottom_right[1] - radius), radius, bubble_color, -1)
        cv2.circle(image, (bottom_right[0] - radius, bottom_right[1] - radius), radius, bubble_color, -1)

        # Draw the tail of the speech bubble, from right half of lower boarder to left side of mouth
        tail_points = np.array([
                [bottom_right[0] - int(bubble_size[0] / 3) , bottom_right[1]], 
                [bottom_right[0] + radius - int(bubble_size[0] / 3), bottom_right[1]], 
                [bubble_x, bubble_y]], np.int32)

        cv2.fillPoly(image, [tail_points], bubble_color)

        # Calculate text size to center it within the bubble
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = top_left[0] + (bubble_size[0] - text_size[0]) // 2
        text_y = top_left[1] + (bubble_size[1] + text_size[1]) // 2

        # Add text to the bubble
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)