import cv2
import numpy as np 
import time
from numpy import *
from rubik_solver import utils
color_code = {"organe": [7,199,77,15,255,255], "blue": [95,190,20,140,255,255],"red": [0,200,70,6,255,255], "yellow": [26,170,75,35,255,255], "green": [60,100,30,96,255,255],"white": [0,0,130,255,42,255]}
# color_code = {"organe": [10,201,83,22,255,255], "blue": [90,150,55,140,255,255],"red": [0,35,54,7,255,255], "yellow": [20,198,84,36,255,255],"green": [35,63,0,75,255,255]}
color_box = {"organe": (0,127,255), "blue":(255,0,0) ,"red":(0,0,255) , "yellow":(0,255,255) ,"green":(0,255,0),"white":(255,0,255)}
color_order = {"organe": 0, "blue":1 ,"red":2 , "yellow":3 ,"green":4,"white":5}
move_sample = ["U","U'","F","F'","R","R'","L","L'","D","D'","B","B'","M","M'","E","E'","S","S'","X","X'","Y","Y'","Z","Z'","U2","F2","R2","L2","D2","B2"]

def remove_small_contours(image):
    try:
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(image, image, mask=mask)
        return image_remove
    except:
        return image

def morphology(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    b_img = cv2.inRange(hsv, (0,0,0), (255,255,15))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    b_img = cv2.dilate(b_img,kernel,iterations = 1)
    # b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations =1)
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    # b_img = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    b_img = remove_small_contours(b_img) 
    b_img = cv2.erode(b_img,kernel,iterations = 1)
    b_img = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return b_img

def find_rubik(b_img, img):
    global px,py,pw,ph
    try:
        contours, hierachy = cv2.findContours(b_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            Area = cv2.contourArea(cnt)
            if Area > 10000:
                x,y,w,h = cv2.boundingRect(cnt)
                # rect = cv2.minAreaRect(cnt)
                # x = int(x) - int(w/2)
                # y = int(y) - int(h/2)
                # w = int(w)
                # h = int(h)
                # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)  
                # cv2.putText(img, str(int(Area)),(x,y), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                px = x
                py = y
                pw = w
                ph = h
        return x,y,w,h
    except:
        print("CANNOT FIND CONTOUR")
        return px,py,pw,ph

def color_distinct(img,color_str):
    global list_box
    #filter 
    img_pro = cv2.GaussianBlur(img.copy(), (5,5), 0)
    #tru mau
    hsv = cv2.cvtColor(img_pro, cv2.COLOR_BGR2HSV)
    color = color_code[str(color_str)]
    color_box_show = color_box[str(color_str)]
    b_img = cv2.inRange(hsv, (color[0],color[1],color[2]), (color[3],color[4],color[5]))
    #morphoogy
    if color_str == "white":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=9)
        b_img = cv2.erode(b_img,kernel,iterations = 5)
        d_img_show = b_img.copy()
        d_img = cv2.distanceTransform(b_img, cv2.DIST_L1,3)
        cv2.normalize(d_img,d_img,0, 1.0, cv2.NORM_MINMAX)
        # d_img_show = d_img.copy()
        d_img = cv2.threshold(d_img, 0.3, 1.0, cv2.THRESH_BINARY)[1]
        d_img = cv2.dilate(d_img,kernel,iterations = 4)
        d_img = d_img*255
        d_img = d_img.astype(np.uint8)

    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        b_img = cv2.morphologyEx(b_img, cv2.MORPH_OPEN, kernel, iterations=2)
        b_img = cv2.dilate(b_img,kernel,iterations = 2)
        d_img = cv2.distanceTransform(b_img, cv2.DIST_L1,3)
        cv2.normalize(d_img,d_img,0, 1.0, cv2.NORM_MINMAX)
        d_img_show = d_img.copy()
        d_img = cv2.threshold(d_img, 0.1, 1.0, cv2.THRESH_BINARY)[1]
        d_img = d_img*255
        d_img = d_img.astype(np.uint8)
    #find contours
    contours, hierachy = cv2.findContours(d_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        Area = cv2.contourArea(cnt)
        if Area > 1250:
            x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),color_box_show,2)  
            cv2.circle(img, (x+int(w/2),y+int(h/2)),8,color_box_show,15)
            # cv2.putText(img, str(Area),(x,y), cv2.FONT_HERSHEY_SIMPLEX,0.5,color_box_show,1)
            list_box.append([(x+int(w/2)),(y+int(h/2)),color_str])
    
    return b_img, d_img, d_img_show




def draw_tutorial(img,state,w,h):
    x1,y1 = int(x + w/6), int(y + h/6)
    x2,y2 = int(x + 3*w/6), int(y + h/6)
    x3,y3 = int(x + 5*w/6), int(y + h/6)
    x4,y4 = int(x + w/6), int(y + 3*h/6)
    x5,y5 = int(x + 3*w/6), int(y + 3*h/6)
    x6,y6 = int(x + 5*w/6), int(y + 3*h/6)
    x7,y7 = int(x + w/6), int(y + 5*h/6)
    x8,y8 = int(x + 3*w/6), int(y + 5*h/6)
    x9,y9 = int(x + 5*w/6), int(y + 5*h/6)
    
    # cv2.line(img_1, (), (),(0,255,0), 3)

    if state == "U": #U
        cv2.line(img_1, (x1,y1), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1-30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)
        

    elif state == "U'":#U'
        cv2.line(img_1, (x1,y1), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3-30), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3+30), (x3,y3),(0,255,0), 3)

    elif state == "F":#F
        cv2.line(img_1, (x8,y8), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4,y4), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2,y2), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6,y6-30), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6-30,y6), (x6,y6),(0,255,0), 3)


    elif state == "F'":#F'
        cv2.line(img_1, (x8,y8), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x4,y4), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2,y2), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x4,y4-30), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4), (x4,y4),(0,255,0), 3)

    elif state == "R":#R
        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3+30), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3+30,y3+30), (x3,y3),(0,255,0), 3)
        
    elif state == "R'":#R'
        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9-30), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9+30,y9-30), (x9,y9),(0,255,0), 3)

    elif state == "L":#L
        cv2.line(img_1, (x1,y1), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7-30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7-30), (x7,y7),(0,255,0), 3)

    elif state == "L'":#L'
        cv2.line(img_1, (x1,y1), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x1-30,y1+30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)

    elif state == "D":#D
        cv2.line(img_1, (x9,y9), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9-30), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9+30), (x9,y9),(0,255,0), 3)

    elif state == "D'":#D'
        cv2.line(img_1, (x9,y9), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7+30), (x7,y7),(0,255,0), 3)

    elif state == "B":#B
        cv2.line(img_1, (x7,y7), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3,y3), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1-30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)

    elif state == "B'":#B'
        cv2.line(img_1, (x7,y7), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x7,y7), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x3,y3), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3-30), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3+30), (x3,y3),(0,255,0), 3)

    elif state == "M":#M
        cv2.line(img_1, (x2,y2), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8-30,y8-30), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8+30,y8-30), (x8,y8),(0,255,0), 3) 

    elif state == "M'":#M'
        cv2.line(img_1, (x2,y2), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x2-30,y2+30), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2+30,y2+30), (x2,y2),(0,255,0), 3)

    elif state == "E":#E
        cv2.line(img_1, (x4,y4), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6-30,y6-30), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6-30,y6+30), (x6,y6),(0,255,0), 3)
 
    elif state == "E'":#E'
        cv2.line(img_1, (x4,y4), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4-30), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4+30), (x4,y4),(0,255,0), 3)

    elif state == "S":#S
        cv2.circle(img_1,(x5,y5), x6-x5,(0,255,0),3)
        cv2.line(img_1, (x6-20,y6-20), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6+20,y6-20), (x6,y6),(0,255,0), 3)

    elif state == "S'":#S'
        cv2.circle(img_1,(x5,y5), x6-x5,(0,255,0),3)
        cv2.line(img_1, (x4-20,y4-20), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4+20,y4-20), (x4,y4),(0,255,0), 3)

    elif state == "X":#X
        cv2.line(img_1, (x1,y1), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x1-30,y1+30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)

        cv2.line(img_1, (x2,y2), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x2-30,y2+30), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2+30,y2+30), (x2,y2),(0,255,0), 3)

        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3+30), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3+30,y3+30), (x3,y3),(0,255,0), 3)

    elif state == "X'":#X'
        cv2.line(img_1, (x1,y1), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7-30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7-30), (x7,y7),(0,255,0), 3)

        cv2.line(img_1, (x2,y2), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8-30,y8-30), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8+30,y8-30), (x8,y8),(0,255,0), 3)

        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9-30), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9+30,y9-30), (x9,y9),(0,255,0), 3)

    elif state == "Y":#Y
        cv2.line(img_1, (x1,y1), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1-30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)

        cv2.line(img_1, (x4,y4), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4-30), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4+30), (x4,y4),(0,255,0), 3)

        cv2.line(img_1, (x7,y7), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7+30), (x7,y7),(0,255,0), 3)
    elif state == "Y'":#Y'
        cv2.line(img_1, (x1,y1), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3-30), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3+30), (x3,y3),(0,255,0), 3)

        cv2.line(img_1, (x4,y4), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6-30,y6-30), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6-30,y6+30), (x6,y6),(0,255,0), 3)

        cv2.line(img_1, (x7,y7), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9-30), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9+30), (x9,y9),(0,255,0), 3)
    elif state == "Z":#Z
        cv2.line(img_1, (x4,y4), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2,y2), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6,y6), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8,y8), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x8+30,y8), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8,y8-30), (x8,y8),(0,255,0), 3)
    elif state == "Z'":#Z'
        cv2.line(img_1, (x4,y4), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2,y2), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6,y6), (x8,y8),(0,255,0), 3)
        cv2.line(img_1, (x8,y8), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4,y4-30), (x4,y4),(0,255,0), 3)
    elif state == "U2":#U2
        cv2.line(img_1, (x1,y1), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1-30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x2,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    elif state == "F2":#F2
        cv2.line(img_1, (x8,y8), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4,y4), (x2,y2),(0,255,0), 3)
        cv2.line(img_1, (x2,y2), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6,y6-30), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x6-30,y6), (x6,y6),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x2,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    elif state == "R2":#R2
        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3-30,y3+30), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x3+30,y3+30), (x3,y3),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x6+10,y6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
    elif state == "L2":#L2
        cv2.line(img_1, (x1,y1), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7-30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x4-25,y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    elif state == "D2":#D2
        cv2.line(img_1, (x9,y9), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9-30), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x9-30,y9+30), (x9,y9),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x8,y8+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    elif state == "B2":#B2
        cv2.line(img_1, (x7,y7), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3,y3), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x3,y3), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1-30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x6+10,y6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    elif state == "Y2":#Y
        cv2.line(img_1, (x1,y1), (x3,y3),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1-30), (x1,y1),(0,255,0), 3)
        cv2.line(img_1, (x1+30,y1+30), (x1,y1),(0,255,0), 3)

        cv2.line(img_1, (x4,y4), (x6,y6),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4-30), (x4,y4),(0,255,0), 3)
        cv2.line(img_1, (x4+30,y4+30), (x4,y4),(0,255,0), 3)

        cv2.line(img_1, (x7,y7), (x9,y9),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7-30), (x7,y7),(0,255,0), 3)
        cv2.line(img_1, (x7+30,y7+30), (x7,y7),(0,255,0), 3)
        cv2.putText(img_1, str(2), (x2,y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    else:
        print("SATE khong hop le")



vid = cv2.VideoCapture(2)
prev_frame_time = 0
new_frame_time = 0
solve = False
color_rubik = np.zeros((6,3,3), dtype=int)
solving_start = False
i = 0
c_time = 0
p_time = 0
temp_solve = False
px,py,pw,ph = 1,1,1,1

while(True):
    ret, img = vid.read()
    
    if not(solve):
    #===================FIND RUBIK CONTOUR=============================
        b_img = morphology(img)
        x,y,w,h = find_rubik(b_img, img)
        rubik_img = img[y:y+h,x:x+w]
        count = 0
    #==================FIND COLOR COORDINATE===========================
        list_box = []
        for i in color_code:
            b_img_color , d_img, d_img_show = color_distinct(rubik_img, str(i))

    #==================TOA DO MAU======================================
        try:
            top = np.zeros((3,3), dtype=int)
            hang_1 = []
            hang_2 = []
            hang_3 = []
            if len(list_box) == 9:
                for i, color in enumerate(list_box):
                    # print("x=",color[0],"y=",color[1],"color:",color[2])
                    if color[1] > 0 and color[1] < int(w/3):
                        hang_1.append(list_box[i])
                    if color[1] > int(w/3) and color[1] < int(2*w/3):
                        hang_2.append(list_box[i])
                    if color[1] > int(2*w/3):
                        hang_3.append(list_box[i])
                
                hang_1 = np.asarray(hang_1)
                hang_2 = np.asarray(hang_2)
                hang_3 = np.asarray(hang_3)
                hang  = [hang_1,hang_2,hang_3]
                for j,row  in enumerate(hang):
                    top[j,0] = color_order[row[np.argmin(np.array(row[:,0], dtype=int)),2]]
                    top[j,2] = color_order[row[np.argmax(np.array(row[:,0], dtype=int)),2]]
                    da_chon_1 =np.argmin(np.array(row[:,0], dtype=int)) + np.argmax(np.array(row[:,0], dtype=int))
                    if da_chon_1 == 1:
                        top[j,1] = color_order[row[2,2]]
                    if da_chon_1 == 2:
                        top[j,1] = color_order[row[1,2]]
                    if da_chon_1 == 3:
                        top[j,1] = color_order[row[0,2]]
                # print("COLOR: ",top)
                print("PLEASE CHOOSE SURFACE")
                if top[1,1] == 3:
                    color_rubik[0,:,:] = top
                    print("TOP PLANE: ", color_rubik[0,:,:])
                if top[1,1] == 1:
                    color_rubik[1,:,:] = top
                    print("LEFT: ", color_rubik[1,:,:])
                if top[1,1] == 2:
                    color_rubik[2,:,:] = top
                    print("FRONT: ", color_rubik[2,:,:])
                if top[1,1] == 4:
                    color_rubik[3,:,:] = top
                    print("RIGHT: ", color_rubik[3,:,:])
                if top[1,1] == 0:
                    color_rubik[4,:,:] = top
                    print("BEHIND: ", color_rubik[4,:,:])
                if top[1,1] == 5:
                    color_rubik[5,:,:] = top
                    print("BOT: ", color_rubik[5,:,:])
                # if cv2.waitKey(1) & 0xFF == ord('l'):
                    solve = True
                    solving_start = True
                    temp_solve = True
                    p_time = time.time()
                    define_color = {'0': 'o', '1':'b','2': 'r', '3':'y','4': 'g', '5':'w'}
                    print("SOLVING!!!")
                    print(color_rubik)
                    cube_str = ''
                    for i in color_rubik:
                        for j in i:
                            for k in j:
                                cube_str += define_color[str(k)]
                                # print(define_color(str(k)))
                    # cube_str = 'wowgybwyogygybyoggrowbrgywrborwggybrbwororbwborgowryby'
                    solver = utils.solve(cube_str, 'Kociemba')
                    solver = np.array(solver)
                    limit = len(solver)
                    i = 0
                    p_time = time.time()
                    
                    
            else:        
                # print("NOT ENOUGH N0 COLOR")
                pass
        except:
            # print("CANNOT GET COLOR COORDINATE")
            pass
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        
        cv2.putText(img, "FPS:"+str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("img", img)
        # cv2.imshow("binary", b_img_color)
        # cv2.imshow("distance", d_img)
        # cv2.imshow("rubik", rubik_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("rubik_2.png", rubik_img)
            break

    else:
        state = ""
        if solving_start:
            
            # ret, img_1 = vid.read()
            img_1 = img.copy()
            b_img_1 = morphology(img_1)

            x,y,w,h = find_rubik(b_img_1, img_1)
            rubik_img_1 = img[y:y+h,x:x+w]
            w = rubik_img_1.shape[1]
            h = rubik_img_1.shape[0]
            
            state = str(solver[i])
            print(state)
            
            draw_tutorial(img_1, state,w,h)
             
            # cv2.imshow("rubik", rubik_img_1) 
            c_time = time.time()
            diff = c_time - p_time
            if diff >= 3:
                i+=1 
                p_time = time.time() 
                
            cv2.putText(img_1, str(round(diff,2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA) 
            cv2.putText(img_1, state, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA) 
            cv2.putText(img_1, str(i+1)+"/"+str(limit), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 
            # cv2.putText(img_1, "Con lai:" +str(limit-i), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 
            cv2.imshow("img_1", img_1) 
            if i >= limit:
                print("======================================WELL DONE============================================")   
                break
            
        else:
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            
            cv2.putText(img, "FPS:"+str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("img", img)
            # cv2.imshow("binary", b_illlll_img)
            # cv2.imshow("rubik", rubik_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite("rubik_2.png", rubik_img)
            break
        
vid.release()
cv2.destroyAllWindows()

