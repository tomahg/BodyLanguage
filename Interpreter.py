# Brainfuck interpreter
# Does not implement input (,)
# The input code split over multiple lines with space
# Each call to step returns the char index + line index of a non-space command + any output
# When the end of the program is reaced (False, c, l, '') is returned

import cv2
import numpy as np

class Visualnterpreter:
    INTERPRETER_OFFSET_Y = 0

    code = []
    jumpmap = {}
    cells = []
    cell_pointer = 0
    code_pointer_char = 0
    code_pointer_line = 0
    finished = False
    debug_slowdown_count = 0
    debug_slowdown_factor = 3
    in_loop_level = 0

    def build_jumpmap(self):
        temp_jumpstack, jumpmap = [], {}
        for line_number, code_line in enumerate(self.code):
            for char_number, command in enumerate(code_line):
                if command == '[': 
                    temp_jumpstack.append((char_number, line_number))
                if command == ']':
                    if len(temp_jumpstack) == 0:
                        print('Syntax error: Can\'t close a loop that hasn\'t started yet!')
                        return False
                    start = temp_jumpstack.pop()
                    jumpmap[start] = (char_number, line_number)
                    jumpmap[(char_number, line_number)] = start
        if len (temp_jumpstack) > 0:
            print('Syntax error: Unclosed loop detected!')
            return False
        self.jumpmap = jumpmap
        return True

    def input_code(self, code):
        self.code = code

    def prepare_code(self):
        ok = self.build_jumpmap()
        if not ok:
            return False
        self.cells = []
        self.cell_pointer = 0
        self.code_pointer_char = 0
        self.code_pointer_line = 0
        self.debug_slowdown_count = 0
        self.finished = False
        self.in_loop_level = 0
        return True

    def step(self):
        char = self.code_pointer_char
        line = self.code_pointer_line
        output = ''

        if self.in_loop_level == 0:
            if self.debug_slowdown_count % self.debug_slowdown_factor != 0:
                self.debug_slowdown_count += 1
                return False, char, line, '' 
            else:
                self.debug_slowdown_count = 1

        # No code, or point past last line
        if len(self.code) == 0:
            print('No code to execute')
            return True, char, line, ''
        elif len(self.code) == self.code_pointer_line:
            return True, char, line, ''

        command = self.code[self.code_pointer_line][self.code_pointer_char]
        # Handle formatted with single spaces, never double
        if command == ' ':
            self.code_pointer_char += 1
            char = self.code_pointer_char
            command = self.code[self.code_pointer_line][self.code_pointer_char]

        if command == ">":
            self.cell_pointer += 1           

        # Dynamically add more cells as needed
        if self.cell_pointer == len(self.cells):
            self.cells.append(0)

        if command == "<":
            if self.cell_pointer <= 0:
                self.cell_pointer = 0
            else: 
                self.cell_pointer -= 1

        if command == "+":
            self.cells[self.cell_pointer] += 1 
            if self.cells[self.cell_pointer] > 255:
                self.cells[self.cell_pointer] = 0

        if command == "-":
            self.cells[self.cell_pointer] -= 1
            if self.cells[self.cell_pointer] < 0:
                self.cells[self.cell_pointer] = 255

        if command == "[":
            if self.cells[self.cell_pointer] == 0: 
                self.code_pointer_char, self.code_pointer_line = self.jumpmap[(self.code_pointer_char, self.code_pointer_line)]
            else:
                self.in_loop_level += 1

        if command == "]":
            if self.cells[self.cell_pointer] != 0: 
                self.code_pointer_char, self.code_pointer_line = self.jumpmap[(self.code_pointer_char, self.code_pointer_line)]
            else:
                self.in_loop_level -= 1

        if command == ".": 
            output = chr(self.cells[self.cell_pointer])

        if self.code_pointer_char < len(self.code[self.code_pointer_line]) - 1:
            self.code_pointer_char += 1
        elif self.code_pointer_line < len(self.code):
            self.code_pointer_char = 0
            self.code_pointer_line += 1

        return False, char, line, output

    def print_single_line_of_code(self, img, line_number, line_of_code, margin_h, color = (255,255,255)):
        line_height = 36
        line_margin_v = 8
        cv2.rectangle(img, (0, line_number * line_height), (img.shape[1], line_height + line_number * line_height), (0,0,0), -1)
        cv2.putText(img, line_of_code.strip(), (margin_h, line_number * line_height + (line_height - line_margin_v)), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    def print_lines_of_code(self, img, n, margin_h):
        for i, line_of_code in enumerate(self.code[-n:]):
            self.print_single_line_of_code(img, i, line_of_code, margin_h)

    def get_text_width(self, text, font_face, font_scale, font_line_thickness):
        ((txt_w, _), _) = cv2.getTextSize(text, font_face, font_scale, font_line_thickness)
        return txt_w
    
    def get_text_size(self, text, font_face, font_scale, font_line_thickness):
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font_face, font_scale, font_line_thickness)
        return txt_w, txt_h    

    def draw_black_alpha_box(self, img, x, y, h, w):
        sub_img = img[y:y+h, x:x+w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.3, black_rect, 0.7, 1.0)
        img[y:y+h, x:x+w] = res

    def debug_single_line_of_code(self, img, line_number, line_of_code, margin_h, margin_v, color = (255,255,255)):
        line_height = 36
        line_margin_v = 8
        cv2.putText(img, line_of_code.strip(), (margin_h, line_number * line_height + margin_v + (line_height - line_margin_v)), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    def debug_lines_of_code(self, img, margin_h):
        if len(self.code) == 0:
            return
        lines = len(self.code)
        self.draw_black_alpha_box(img, 0, self.INTERPRETER_OFFSET_Y, 40 * lines, img.shape[1])
        for i, line_of_code in enumerate(self.code):
            self.debug_single_line_of_code(img, i, line_of_code, margin_h, self.INTERPRETER_OFFSET_Y)

    def highlight_debug_command(self, img, char_number, line_number, margin_h, color = (50, 205, 50)):
        if line_number == len(self.code):
            return
        line_height = 36
        line_margin_v = 8
        previous_code = self.code[line_number][:char_number]
        if len(previous_code) > 0:
            # Subtract 2, beacuse the measurement of an empty string apparently is 2
            offset = self.get_text_width(previous_code, cv2.FONT_HERSHEY_PLAIN, 2, 2) - 2
        else:
            offset = 0
        command = self.code[line_number][char_number]
        cv2.putText(img, command, (margin_h + offset, self.INTERPRETER_OFFSET_Y + line_number * line_height + (line_height - line_margin_v)), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    # Print the first 8 cells
    #                   v
    # +-------------------------------+
    # | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 
    # +-------------------------------+
    def print_cells(self, img):
        self.draw_black_alpha_box(img, 0, 428, 70, img.shape[1])
        
        # Draw cells
        cv2.line(img, (4, 432), (636, 432), (255,255,255), 2)
        cv2.line(img, (4, 476), (636, 476), (255,255,255), 2)
        for i in range(4, 640, 79):
            cv2.line(img, (i, 432), (i, 476), (255,255,255), 2)
        
        # Draw cell values
        for i, cell in enumerate(self.cells[:8]):
            offset = self.get_text_width(str(cell), cv2.FONT_HERSHEY_PLAIN, 2, 2) - 2
            cv2.putText(img, str(cell), ((i + 1) * 79 - offset, 465), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            
        # Draw pointer to current cell (cell pointer)
        xp = 40 + 79 * self.cell_pointer
        yp = 422
        cv2.line(img, (xp, yp), (xp - 20, yp - 15), (255,255,255), 5)
        cv2.line(img, (xp, yp), (xp + 20, yp - 15), (255,255,255), 5)        
        cv2.line(img, (xp, yp), (xp - 20, yp - 15), (0,0,255), 3)
        cv2.line(img, (xp, yp), (xp + 20, yp - 15), (0,0,255), 3)

    def print_outout(self, img, output):
        if len(output) > 0:
            font_size = 5
            font_thickness = 3
            width, height = self.get_text_size(output, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)
            
            # If text is too long, make it smaller - still only single line
            while width > img.shape[1] - 10:
                if font_size > 2:
                    font_size -= 1
                else:
                    font_size *= 0.75
                if font_thickness > 1:
                    font_thickness -= 1
                width, height = self.get_text_size(output, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)

            if width > 0 and height > 0:
                # Consider allowing multipe lines of output
                self.draw_black_alpha_box(img, 0, 320, 70, img.shape[1])      
                # Offset to keep text centered, wile getting smaller
                offset = int((47 - height) / 2)    
                cv2.putText(img, output, (int((img.shape[1] - width) / 2) , 380 - offset), cv2.FONT_HERSHEY_PLAIN, font_size, (255,255,255), font_thickness)
