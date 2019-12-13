import tkinter as tk
from keras.models import load_model
from collections import deque
import numpy as np
import cv2
import Pmw

global c
c  =0

def start():
    global C, M, B, Q
    global c

    C = tk.Button(mainWindow, text="Display CNN text file", command=cnn_disp)
    M = tk.Button(mainWindow, text="Display MLP text file", command=mlp_disp)
    Q = tk.Button(mainWindow, text="Quit", command=quit)
    B = tk.Button(mainWindow, text="Start writing again", command=start)

    if c == 0:
        B.place_forget()
        C.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
        C.pack()
        C.focus()
        M.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
        M.pack()
        M.focus()
        B.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
        B.pack()
        B.focus()
        Q.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
        Q.pack()
        Q.focus()
        c += 1

    if c != 0 :
        B.place_forget()
        Q.place_forget()
        M.place_forget()
        C.place_forget()


    mlp_model = load_model('emnist_mlp_model.h5')
    cnn_model = load_model('emnist_cnn_model.h5')

    cnn_file = open('cnn_file.txt', 'w')
    mlp_file = open('mlp_file.txt', 'w')

    letters = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
               11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
               21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

    blueLower = np.array([100, 60, 60])
    blueUpper = np.array([140, 255, 255])

    # erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    s1 = ""
    s2 = ""

    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    alphabet = np.zeros((200, 200, 3), dtype=np.uint8)

    points = deque(maxlen=512)

    prediction1 = 26
    prediction2 = 26

    flag1 = False
    flag2 = False

    index = 0

    camera = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        blueMask = cv2.erode(blueMask, kernel, iterations=2)  # function for dilation and elation
        blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
        blueMask = cv2.dilate(blueMask, kernel, iterations=1)

        (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:

            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Get the moments to calculate the center of the contour (in this case Circle)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            points.appendleft(center)

        elif len(cnts) == 0:
            if len(points) != 0:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = sorted(blackboard_cnts, key=cv2.contourArea, reverse=True)[0]

                    if cv2.contourArea(cnt) > 1000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        alphabet = blackboard_gray[y - 10:y + h + 10, x - 10:x + w + 10]
                        newImage = cv2.resize(alphabet, (28, 28))
                        newImage = np.array(newImage)
                        newImage = newImage.astype('float32') / 255

                        prediction1 = mlp_model.predict(newImage.reshape(1, 28, 28))[0]
                        prediction1 = np.argmax(prediction1)
                        flag1 = True;

                        prediction2 = cnn_model.predict(newImage.reshape(1, 28, 28, 1))[0]
                        prediction2 = np.argmax(prediction2)
                        flag2 = True;

                # Empty the points deque and the blackboard
                points = deque(maxlen=512)
                blackboard = np.zeros((480, 640, 3), dtype=np.uint8)

        # Connect the points with a line
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)
            cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)

        k1 = str(letters[int(prediction1) + 1])
        k2 = str(letters[int(prediction2) + 1])

        if (flag1):
            s1 += k1

        if (flag2):
            s2 += k2

        flag1 = False
        flag2 = False

        cv2.putText(frame, "Multilayer Perceptron : " + s1, (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)
        cv2.putText(frame, "Convolution Neural Network:  " + s2, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        cv2.imshow("alphabets Recognition Real Time", frame)

        k = cv2.waitKey(1)

        if k == 32:
            mlp_file.write(s1)
            x
            cnn_file.write(s2)
            s1 = " "
            s2 = " "
            mlp_file.write(s1)
            cnn_file.write(s2)

        if k == 113:
            mlp_file.write(s1)
            cnn_file.write(s2)
            break;

    camera.release()
    cv2.destroyAllWindows()

def quit():
    exit()

def cnn_disp():
    filename = "cnn_file.txt"
    text = Pmw.ScrolledText(mainWindow,
                            borderframe=5,
                            vscrollmode='dynamic',
                            hscrollmode='dynamic',
                            labelpos='n',
                            label_text='file %s' % filename,
                            text_width=40,
                            text_height=4,
                            text_wrap='none',
                            )
    text.pack()
    text.insert('end', open(filename, 'r').read())


def mlp_disp():
    filename = "mlp_file.txt"
    text = Pmw.ScrolledText(mainWindow,
                            borderframe=5,
                            vscrollmode='dynamic',
                            hscrollmode='dynamic',
                            labelpos='n',
                            label_text='file %s' % filename,
                            text_width=40,
                            text_height=4,
                            text_wrap='none',
                            )
    text.pack()
    text.insert('end', open(filename, 'r').read())


mainWindow = tk.Tk(screenName = "Writing in Air")
mainWindow.resizable(width=False, height=False)

B = tk.Button(mainWindow, text ="Start", command = start)

B.pack()
B.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
B.focus()

mainWindow.mainloop()