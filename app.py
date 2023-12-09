import dxcam
import cv2 as cv
import numpy as np
from time import time
import time as waktu
import os
import pyautogui as pygui
import keyboard
import mouse

# mouse.move("500", "500")
# mouse.click() # default to left click
# mouse.right_click()
# mouse.double_click(button='left')
# mouse.double_click(button='right')
# mouse.press(button='left')
# mouse.release(button='left')

screen_width, screen_height = pygui.size()
x_coor = 0
y_coor = 0
z_coor = 0

EXIT_BUTTON = 'g'
STOPALL_BUTTON = 'h'
START_BUTTON = 'j'
START_FISHING = 'k'
START_MINING = 'l'

active = False
isFishing = False
isMining = False

errVal=10
walkSpd = 4.2
thisdir = os.getcwd()
listNumber = []
listCursor = []
coordinate_arrImg = []
cursor_matchingImg = []
template_position_img = cv.imread('coordinate_asset/position_img.png')
template_position_height, template_position_width = template_position_img.shape[:-1]
template_fisnging_img = cv.imread('fishing_asset/fisnging_rod.jpeg')
template_fisnging_height, template_fishing_width = template_fisnging_img.shape[:-1]
logText=""
comaTresshold = .6
threshold = .8

def screenSize(width = 0.5, height = 0.5, screenWidth=screen_width, screenHeight=screen_height):
    width = width * screenWidth
    height = height * screenHeight
    x1 = int ((screenWidth - width)/2)
    y1 =  int ((screenHeight - height)/2)
    x2 = int (x1 + width)
    y2 = int (y1 + height)
    region = (x1, y1, x2, y2)
    return region

def keyboardListener(key):
    try:
        if keyboard.is_pressed(f'{key}'):
            return True
        else: return False
    except:
        return False


screen = dxcam.create()
region = screenSize(0.4, 0.7)
screen.start(region=region)

# get number image from directory
for r, d, f in os.walk(thisdir + '\coordinate_asset'):
    for file in f:
        if file.endswith(".jpg"):
            listNumber.append(file)

# get cursor image from directory
for r, d, f in os.walk(thisdir + '\cursor_asset'):
    for file in f:
        if file.endswith("cursor.jpeg"):
            listCursor.append(file)

def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


deltaTime = 0
waitTime = .5
last_torc_coor = [0,0,0]
while True:
    last_coor = [99999,99999,99999]
    start_time = time()
    rawImg = screen.get_latest_frame()
    arrimg = cv.cvtColor(rawImg, cv.COLOR_BGR2RGB)

    mouse_x, mouse_y = pygui.position()
    mousePos = 'mouse_x:' + str(mouse_x).rjust(4) + ' mouse_y:' + str(mouse_y).rjust(4)

    if active == True:
        coordinate_arrImg = []
        cursor_matchingImg = []
        # Coordinate Detection
        positionImg_result = cv.matchTemplate(arrimg, template_position_img, cv.TM_CCOEFF_NORMED)
        pos_loc = np.where(positionImg_result >= threshold)
        for pt in zip(*pos_loc[::-1]):  # Switch collumns and rows
            coor_x1 = pt[0] + template_position_width
            coor_x2 = pt[0] + template_position_width + template_position_width + int(template_position_width * 6/7)
            coor_y1 = pt[1]
            coor_y2 = pt[1] + template_position_height 
            cv.rectangle(arrimg, pt, (pt[0] + template_position_width, pt[1] + template_position_height), (0, 0, 255), 2)
            cv.rectangle(arrimg, (coor_x1, coor_y1), (coor_x2, coor_y2), (0, 255, 0), 2)
            coordinate_arrImg = arrimg[coor_y1:coor_y2, coor_x1:coor_x2]

            cursor_matchingImg = arrimg[coor_y1*4:coor_y2*6, coor_x1*3:coor_x2*2]


        # Coordinate Reading
        coorText = []
        if(coordinate_arrImg != []):
            for filename in listNumber:
                template_number_img = cv.imread("coordinate_asset/"+filename)
                filename = filename.split('.')[0]
                if filename == 'coma':
                    confidence = comaTresshold
                else: confidence = threshold
                temp_num_height, temp_num_width = template_number_img.shape[:-1]
                numberResult = cv.matchTemplate(coordinate_arrImg, template_number_img, cv.TM_CCOEFF_NORMED)
                loc = np.where(numberResult >= confidence)
                for pt in zip(*loc[::-1]):  # Switch collumns and rows
                    if [ (x,y) for x, y in coorText if abs(pt[0] - x) <= errVal and y == filename]:                
                        continue 
                    coorText.append((pt[0], filename))
                    cv.rectangle(coordinate_arrImg, pt, (pt[0] + temp_num_width, pt[1] + temp_num_height), (0, 255, 255), 2)  
                coorText.sort(key=lambda tup: tup[0])
            # print(coorText)
            coor_string = ""
            for x in coorText:
                newChar = x[1]
                if newChar == 'minus':
                    newChar = '-'
                if newChar == 'coma':
                    newChar = ','
                coor_string = coor_string + newChar  
            coor_splitString = coor_string.split(',')
            if len(coor_splitString) == 3:
                print(coor_string)  
                if isInt(coor_splitString[0]) and isInt(coor_splitString[1]) and isInt(coor_splitString[2]):
                    x_coor = int(coor_splitString[0])
                    y_coor = int(coor_splitString[1])
                    z_coor = int(coor_splitString[2])
                    last_coor = [x_coor, y_coor, z_coor]
            elif len(coor_splitString) < 3 : 
                if comaTresshold > 0:
                    comaTresshold = comaTresshold - .001
            elif len(coor_splitString) > 3 : 
                if comaTresshold > 0:
                    comaTresshold = comaTresshold + .001

        # Fishing Detection
        if cursor_matchingImg != [] and isFishing:
            rod_detected = False
            fishing_result = cv.matchTemplate(cursor_matchingImg, template_fisnging_img, cv.TM_CCOEFF_NORMED)
            fishing_loc = np.where(fishing_result > .65)
            for pt in zip(*fishing_loc[::-1]):  # Switch collumns and rows
                rod_detected = True
                cv.rectangle(cursor_matchingImg, pt, (pt[0] + template_fishing_width, pt[1] + template_fisnging_height), (255, 255, 255), 2)
                waitTime = .5
                break
            waitTime = waitTime - deltaTime
            if rod_detected == False and waitTime <=0:
                waitTime = .5
                print('mouse right click')
                mouse.right_click()
        if isMining:
            torchDistance = 7
            if abs(x_coor - last_coor[0] <= 1) and abs(y_coor - last_coor[1] <= 1) and abs(z_coor - last_coor[2] <= 1):
                if abs((last_torc_coor[0]) - (x_coor)) >= torchDistance or abs((last_torc_coor[1]) - (y_coor)) >= torchDistance or abs((last_torc_coor[2]) - int(z_coor)) >= torchDistance:
                    keyboard.release('q')
                    keyboard.release('w')
                    waktu.sleep(0.2)
                    pygui.keyDown('num6')
                    waktu.sleep(0.2)
                    pygui.keyUp('num6')
                    waktu.sleep(0.2)
                    pygui.keyDown('num6')
                    waktu.sleep(0.2)
                    pygui.keyUp('num6')
                    waktu.sleep(0.2)
                    keyboard.press_and_release('1',)
                    waktu.sleep(0.2)
                    keyboard.press_and_release('e')
                    waktu.sleep(0.2)
                    keyboard.press_and_release('2')
                    waktu.sleep(0.2)
                    pygui.keyDown('num4')
                    waktu.sleep(0.2)
                    pygui.keyUp('num4')
                    waktu.sleep(0.2)
                    pygui.keyDown('num4')
                    waktu.sleep(0.2)
                    pygui.keyUp('num4')
                    waktu.sleep(0.2)
                    keyboard.press('q')
                    keyboard.press('w')
                    last_torc_coor = [x_coor, y_coor, z_coor]

    else : cv.waitKey(10)
        
    # logText = comaTresshold
    # Count FPS
    now_time = time()
    deltaTime = (now_time - start_time) 
    fps = 1.0/deltaTime
    # print(f"Frames Per Second : {fps:.2f}")
    cv.rectangle(arrimg, (5, 5), (160*3, 160), (0, 0, 0), -1)
    cv.putText(arrimg, f'g:exit, h:stopAll, j:activate, k:startFishing, l:startMining', (10,20), cv.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    cv.putText(arrimg, f'Detection: {active}, AutoFishing: {isFishing}, AutoMining: {isMining}', (10,40), cv.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    cv.putText(arrimg, f'FPS: {fps:.2f}', (10,60), cv.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    cv.putText(arrimg, f'X:{x_coor}, Y:{y_coor}, Z:{z_coor}', (10,80), cv.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)   
    cv.putText(arrimg, f'{mousePos}', (10,100), cv.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    cv.putText(arrimg, f'Wait Time:{waitTime:.2f}', (10,120), cv.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)

    #Show Result
    cv.imshow("Screen", arrimg)
    if(cursor_matchingImg != []):
        cv.imshow("Coordinate", cursor_matchingImg)



    if keyboardListener(START_BUTTON):
        active = True
    if keyboardListener(STOPALL_BUTTON):
        active = False
        isFishing = False
        isMining = False        
        keyboard.release('q')
        keyboard.release('shift')
        keyboard.release('w')
    if keyboardListener(START_FISHING):
        isFishing = True
    if keyboardListener(START_MINING):
        last_torc_coor = [x_coor, y_coor, z_coor]
        keyboard.press('q')
        keyboard.press('shift')
        keyboard.press('w')
        isMining = True
    if cv.waitKey(10) & keyboardListener(EXIT_BUTTON):
        active = False
        isFishing = False
        isMining = False        
        keyboard.release('q')
        keyboard.release('shift')
        keyboard.release('w')
        break