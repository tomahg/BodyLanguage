# Brainfuck interpreter
# Does not implement input (,)
# The input code split over multiple lines with space
# Each call to step returns the char index + line index of a non-space command + any output
# When the end of the program is reaced (-1, -1, '') is returned

class Visualnterpreter:
    code = []
    jumpmap = {}
    cells = []
    cell_pointer = 0
    code_pointer_char = 0
    code_pointer_line = 0

    def build_jumpmap(self):
        temp_jumpstack, jumpmap = [], {}
        for line_number, code_line in enumerate(self.code):
            for char_number, command in enumerate(code_line):
                if command == '[': 
                    temp_jumpstack.append((char_number, line_number))
                if command == ']':
                    start = temp_jumpstack.pop()
                    jumpmap[start] = (char_number, line_number)
                    jumpmap[(char_number, line_number)] = start
            return jumpmap

    def input_code(self, code):
        self.code = code
        self.jumpmap = self.build_jumpmap()
        self.cells = []
        self.cell_pointer = 0
        self.code_pointer_char = 0
        self.code_pointer_line = 0

    def step(self):
        char = self.code_pointer_char
        line = self.code_pointer_line
        output = ''

        if len(self.code) == 0:
            print('No code to execure')
            return (-1, -1, '')

        command = self.code[self.code_pointer_line][self.code_pointer_char]
        # Handle formatted with single spaces, never double
        if command == ' ':
            self.code_pointer_char += 1
            command = self.code[self.code_pointer_line][self.code_pointer_char]

        if command == ">":
            self.cell_pointer += 1           

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

        if command == "[" and self.cells[self.cell_pointer] == 0: 
            self.code_pointer_char, self.code_pointer_line = self.jumpmap[(self.code_pointer_char, self.code_pointer_line)]
        
        if command == "]" and self.cells[self.cell_pointer] != 0: 
            self.code_pointer_char, self.code_pointer_line = self.jumpmap[(self.code_pointer_char, self.code_pointer_line)]
        
        if command == ".": 
            output = chr(self.cells[self.cell_pointer])

        if self.code_pointer_char < len(self.code[self.code_pointer_line]) - 1:
            self.code_pointer_char += 1
        elif self.code_pointer_line < len(self.code) - 1:
            self.code_pointer_char = 0
            self.code_pointer_line += 1
        else:
            self.code_pointer_char = -1
            self.code_pointer_line = -1

        return char, line, output