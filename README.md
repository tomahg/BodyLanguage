# Bodyfuck – Gesture controlled brainfuck interpreter

#### Introduction

This interpreter let the user input, and debug, brainfuck code in a visual manner using only body gestures – keyboard are completely optional.

##### What is brainfuck

Brainfuck is a an esoteric programming language created by Urban Müller in 1993. When the program starts, you have a array initialized to only zeros. Like this. Some implementations have a fixed array length of 30,000 items. Others, like mine, just adds new cells to the array when it gets full.

```
[0][0][0][0][0][0][0][0][0][0]
```

When the program starts, a cell pointer points to the first cell like this. From now on I will refer to the cell pointed to by the cell pointer as "the current cell".

```
[0][0][0][0][0][0][0][0][0][0]
 ^
```

The programming language brainfuck has only eight commands, all of them is only a simple character.

-   \> : move cell pointer to the right.
-   < : move cell pointer to the left.
-   \+ : increment the value in the current cell. Only 8 bits, so it wraps around from 255 to 0.
-   \- : decrement the value in the current cell. If 0 is decremented, it wraps around to 255.
-   , : input, one character input from standard input. The ASCII value of the character is stored in the current cell.
-   . : output, the ASCII character that corresponds to the number in the current cell is sent to standard output
-   \[ : start a loop, if current cell value is 0, jump past corresponding end of loop
-   \] : end a loop, jump back to corresponding start of loop, if current cell value is not 0

That's all there is to say about [brainfuck](https://esolangs.org/wiki/Brainfuck).

##### What is bodyfuck

Bodyfuck is a gestural extension of brainfuck, created by [Nikolaos Hanselmann in 2009](https://sophia.ooo/projects/bodyfuck/). Based on a YouTube video that demonstrates his usage, and a few articles I have recreated an interpreter for this very strange language.

That's all there is to say about [bodyfuck](https://esolangs.org/wiki/Bodyfuck).

### What I have created

Using the [MediaPipe](https://developers.google.com/mediapipe) library from Google, it was quite easy to interpret body gestures from a video stream running from my laptop. This is on-device machine learning, but it runs quite smooth on my five year old laptop. The library gives me [33 body landmarks](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), and their position, for every frame. I use this information to calculate different moves, like raising.

I use [Open CV](https://opencv.org/) for streaming the video from my webcam, and also to draw on the video stream.

-   For each recognized move, the move is drawn on screen.
-   The complerte brainfuck code is drawn on screen.
-   When running/debugging the code, the current brainfuck character is highlighted.
-   When running/debugging the code, I draw the eight first cells of the datastructure at the bottom of the screen.

#### Commands from the original bodyfuck language that I have implemented

-   Increment current cell: Raise one hand.
-   Decrement current cell: Duck below threshold.
-   Move cell pointer right: Move to the right third of the screen.
-   Move cell pointer left: Moce to the left third of the screen.
-   Start a loop: Stay at the left third of the screen for more than 1 second.
-   End a loop: Stay at the right third of the screen for more than 1 second.
-   Output: Both arms stretched out, horizontaly.
-   Start program: Clap your hands.
-   Delete all code: Clap your hands twice. This is the only way to deltete code, in the original. One small typo, and you'll need to start over.

Notice the input is not implemented, neither by me nor Nikolaos Hanselmann

#### Additional commands I have implemented, to ease the developer experience

-   Double increment: Raises both hands at the same time. Implemented because brainfuck/bodyfuck normally involves a lot of incrementing.
-   Delete single character: Facepalming with your right hand. Implemented because humans make typos!
-   Single step code intepreter when paused: Point to the right with a straight right arm.
-	Single step back when paused: Point to the left with a straight right arm to travel back in time!
-   Fast stepping when paused: Point, then keep the elbow out and crank the forearm around it, the way the hand points, for at least half a turn. The faster you crank, the faster the code steps.
-   Insert mode: Fold your arms across your chest and hold it for a second. Implemented because the original language can only ever add code to the very end, so a single missing character in the middle means typing the whole program again.

### Running the program, and it's two modes

#### The code editor

When the program starts you will see a video stream from your webcam. Any code you input will be shown with black background on the top of the screen. If you type a lot of code, you will see the four last lines. For improved readability five consequtive +'es og -'es will have a space after them. This makes it easier to read a lot of increments in a row, as you can count groups of five.

In the editor you can do the following

-   Input code.
-   Delete last character with the facepalm gesture.
-   Delete all code with the double clap.
-   Enter the interpreter with the single clap.
-   Open insert mode, to add or remove code somewhere other than at the end, by folding your arms across your chest for a second.

#### The interpreter

When the interpreter starts, the code will begin running at variable speed. At the top of the screen you will see the code on a transparent black overlay over the video. The current character will be highlightes, and inside a loop the code will run faster, because it's probably not that interesting to see the same loope execute over, and over, and over. At the bottom of the screen you will see the eight first cells, with their values and the cell pointer.

In the editor you can do the following

-   Pause or restart the code execution with a single clap.
-   Restart the code execution with a single clap, if it's execution has finished.
-   Exit the interpreter (return to the editor) with a double clap.
-   When the interpreter is paused (with a single clap), point to the right (straight arm) to run the code a single step forwards, point to the left (straigh arm) to run the code a single step backwards.
-   When stepping one character at a time gets tedious, keep pointing, hold the elbow out to the side, and crank the forearm around the elbow like a handle. The code keeps stepping the way the hand points, and the faster you crank the faster it steps. Each arm has to turn the way it points – as if the hand rolled a wheel in that direction – so the right arm turns clockwise and the left arm counter-clockwise.

In addition to this, the brave amongst you, can edit the code directly in the interpreter WHILE the code runs! Are you confident in your own coding skills to start executing the program you are halfway finished writing, knowing you will be able to finish your code before the interpreter catches up?

Note: When live coding, the interpreter will halt if your code contains incomplete loops. I.e. an \[ without a matching \], or the other way around.

#### Insert mode

Everything above adds code to the end of the program, which is all the original bodyfuck can do. One wrong character halfway through, and there was nothing for it but to start over. Insert mode puts a caret in the code instead, and commands land there.

Stop where you want to work: clap to pause the running code, then step to the exact character with the pointing gesture, or crank the forearm to travel further. Fold your arms across your chest, hold it for a second, and insert mode opens. A green caret appears, standing just after the character that ran last, and the highlighted command, the cells and the output all disappear – nothing runs while you are editing.

In insert mode

-   Any command gesture inserts that command at the caret. Groups of five +'es and -'es are re-counted around it, so a + dropped into the middle of a long run still reads correctly.
-   Facepalm deletes the character in front of the caret, and the caret follows it back.
-   Point to the right or to the left to move the caret one command that way, and crank the forearm to move it faster – the same gestures that step the code, except that here nothing is executed. The caret goes all the way to the front, so code can be inserted before the very first character.
-   Clap once to leave. You land back in the code editor, and the run is gone: cells cleared, output cleared, back to the start. Clap again to run the edited program.


#### Keyboard commands

For debugging there are some keyboard commands available

-   c: Show/hide the code
-   g: Show grid lines, to calibrate how far to the right/left you'll need to move, and how low you need to duck
-   p: Pause code input
-   backspace: Delete single character
-   delete: Clear all code

Of course, the use of keyboard commands is frowned upon by the bodyfuck community.

#### Creating a virtual environment
'''
python -m venv .venv
pip install -r requirements.txt
'''

#### Running from a virtual environment
'''
.venv\Scripts\activate.bat
python bodyfuck.py
'''

Enjoy!
