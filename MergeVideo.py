import cv2
import numpy as np
import tkinter as tk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk
import time


cropping = False
list_refPt = []


def crop_image(event, x, y, flags, param):
    global refPt, cropping, list_refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [[x, y]]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append([x, y])
        cropping = False
        cv2.rectangle(param, tuple(refPt[0]), tuple(refPt[1]), (0, 255, 255), 1)
        refPt[0][0] += 2
        refPt[0][1] += 2
        refPt[1][0] -= 2
        refPt[1][1] -= 2
        list_refPt.append(refPt)
        cv2.imshow('image', param)


def choose_range(image):
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", crop_image, param=clone)
    while True:
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clone = image.copy()
        elif key == ord("c"):
            break


def roi_image(list_refPt, image):
    list_roi = []
    for refPt in list_refPt:
        clone = image.copy()
        list_roi.append(clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]])
    return list_roi


def gaussian(x):
    muy = np.mean(x)
    sigma = np.std(x)
    low = muy - 2 * sigma
    upper = muy + 2 * sigma
    return abs(int(low)), abs(int(upper))


def find_H(low_lst, high_lst):
    min = 255
    max = 0
    for i in low_lst:
        if i[0] < min:
            min = i[0]
    for j in high_lst:
        if j[0] > max:
            max = j[0]
    return min, max


def find_S(low_lst, high_lst):
    min = 255
    max = 0
    for i in low_lst:
        if i[1] < min:
            min = i[1]
    for j in high_lst:
        if j[1] > max:
            max = j[1]
    return min, max


def find_V(low_lst, high_lst):
    min = 255
    max = 0
    for i in low_lst:
        if i[2] < min:
            min = i[2]
    for j in high_lst:
        if j[1] > max:
            max = j[2]
    return min, max


def find_range(image, list_refPt):
    list_roi = roi_image(list_refPt, image)  # các vùng ảnh
    low_lst = []
    high_lst = []
    for img in list_roi:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Tách vùng H, S, V
        h, s, v = cv2.split(img_hsv)

        # Tìm min, max H, S, V
        low_h, high_h = gaussian(h)
        low_s, high_s = gaussian(s)
        low_v, high_v = gaussian(v)

        low_lst.append([low_h, low_s, low_v])
        high_lst.append([high_h, high_s, high_v])
    return low_lst, high_lst


def chroma_pro(image, background, low_lst, high_lst):
    min, max = find_H(low_lst, high_lst)
    # S, V thường min max 100,255 20,255

    LOW = np.array([min - 10, 100, 50])
    HIGH = np.array([max + 10, 255, 255])

    # Test thu nghiem, chuyen qua anh HSV
    obj = image
    bg = background
    bg = cv2.resize(bg, (obj.shape[1], obj.shape[0]), interpolation=cv2.INTER_AREA)
    obj_hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)

    # Tìm mask object và background
    mask_bg = cv2.inRange(obj_hsv, LOW, HIGH)
    mask_obj = cv2.bitwise_not(mask_bg, mask=None)

    kernel = np.ones((5, 5), np.uint8)
    mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel)
    mask_bg = cv2.bitwise_not(mask_obj, mask=None)

    # Ảnh object và background
    image_obj = cv2.bitwise_and(obj, obj, mask=mask_obj)

    image_bg = cv2.bitwise_and(bg, bg, mask=mask_bg)

    chroma_image = cv2.bitwise_or(image_obj, image_bg)
    # cv2.imshow("chroma_image", chroma_image)
    # cv2.waitKey(0)
    return chroma_image


def image_process(file_img, file_bgr):
    img = cv2.imread(file_img)
    bgr = cv2.imread(file_bgr)
    choose_range(img)
    low_lst, high_lst = find_range(img, list_refPt)
    chroma_img = chroma_pro(img, bgr, low_lst, high_lst)
    cv2.imshow('Result', chroma_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_process(file_video, file_bgr):
    cap = cv2.VideoCapture(file_video)
    ret, frame = cap.read()
    bgr = cv2.imread(file_bgr)
    choose_range(frame)
    low, high = find_range(frame, list_refPt)
    frame = chroma_pro(frame, bgr, low, high)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(file_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (1280, 720))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            # start = time.time()
            frame = chroma_pro(frame, bgr, low, high)
            # print(time.time() - start)

            cv2.imshow('frame', frame)
            if cv2.waitKey(int(100/fps)) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Processing...')

    i = 0
    cap = cv2.VideoCapture(file_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            frame = chroma_pro(frame, bgr, low, high)
            out.write(frame)
            i += 1
            print('Frame', i)
        else:
            break
    print('Done')
    cap.release()
    cv2.destroyAllWindows()


def live_process(file_bgr):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    bgr = cv2.imread(file_bgr)
    flags = True
    frame = None
    while flags:
        ret, frame = cap.read()
        cv2.imshow("Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("z"):
            choose_range(frame)
            flags = False
            break
    low_lst, high_lst = find_range(frame, list_refPt)
    # find range of H value
    min_h, max_h = find_H(low_lst, high_lst)

    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)q
    cap.release()
    cv2.destroyAllWindows()

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.resize(frame, (640, 480))
        bgr = cv2.resize(bgr, (640, 480))

        low_green = np.array([min_h - 10, 100, 20])
        high_green = np.array([max_h + 10, 255, 255])

        mask = cv2.inRange(hsv_frame, low_green, high_green)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # background mask1  # xoa nen ngoai thay bang mau den
        mask_1 = np.copy(mask)

        # mask_2          # xoa nen xanh thay bang mau den
        mask_2 = np.copy(frame)
        mask_2[mask != 0] = [0, 0, 0]

        # mask_3
        mask_3 = np.copy(mask_2)
        mask_3[mask == 0] = [255, 255, 255]

        # show background
        f = frame - res
        f = np.where(f == 0, bgr, f)

        # show result
        # cv2.imshow('reality camera', frame)
        cv2.imshow('frame', frame)
        cv2.imshow("mask1 ", mask)
        cv2.imshow('mask2', mask_2)
        cv2.imshow('res', res)
        cv2.imshow('background', f)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def close_window():
    root.destroy()

file_img = 'image_video/human.jpg'
file_bgr = 'image_video/BG_1.jpg'
file_video ='image_video/Green_effect_1.mp4'


root = tk.Tk()
root.geometry('720x360')

subBG = Image.open('image_video/BG.jpg')
subBG.resize((720, 360))
subBG = ImageTk.PhotoImage(subBG)

frameMain = Frame(root)
frameMain.pack()

varMain = StringVar()
labelMain = Label(frameMain, textvariable=varMain, relief=RAISED, height=100, width=720, padx=20, pady=20)
labelMain.config(font=("Courier", 16, "bold"))
labelMain.config(image=subBG, compound='center')
varMain.set("Welcome to our project\nRemove background with Chroma Key")

varSub = StringVar()
labelSub = Label(frameMain, textvariable=varSub, relief=RAISED, bd=0)
labelSub.config(font=("Courier", 12), pady=10)
varSub.set("Processing with: ")

labelMain.pack()
labelSub.pack()

frameSub = Frame(root)
frameSub.pack()

frameBottom = Frame(root)
frameBottom.pack(side=BOTTOM, fill='x')


A = Button(frameSub, text='Image', command=image_process(file_img, file_bgr), padx=10, pady=10)
A.config(font=("Courier", 12))
list_refPt = []
B = Button(frameSub, text='Video', command=video_process(file_video, file_bgr), padx=10, pady=10)
B.config(font=("Courier", 12))
C = Button(frameSub, text='Live', command=live_process(file_bgr), padx=10, pady=10)
C.config(font=("Courier", 12))
list_refPt = []
D = Button(frameBottom, text='End', command=close_window, padx=10, pady=10)
D.config(font=("Courier", 12))

A.grid(column=0, row=0, padx=10, pady=20)
B.grid(column=1, row=0, padx=10, pady=20)
C.grid(column=3, row=0, padx=10, pady=20)
D.pack(side=RIGHT)
root.mainloop()



