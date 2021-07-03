import cv2
import numpy as np
import math

# Open webcam
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # read image
    ret, img = cap.read()

    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (100,100), (500,500), (0,255,0),0)
    crop_img = img[100:500, 100:500]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur (remove sharp edges)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholding: Otsu's Binarization 
    # (separate pixels in background and foreground)
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # show thresholded image
    cv2.namedWindow('Thresholded', cv2.WINDOW_NORMAL)
    cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '4':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # find contour with max area (biggest foreground object)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(crop_img,start, end, [0,255,0], 2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)

    # define actions required
    '''
    if count_defects == 0:
        cv2.putText(img,"This means that we could detect a fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 1:
        cv2.putText(img,"This means that we could detect 1 finger", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 2:
        str = "This means that we could detect 2 fingers"
        cv2.putText(img, str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 3:
        cv2.putText(img,"This means that we could detect 3 fingers", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 4:
        cv2.putText(img,"This means that we could detect 4 fingers", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    else:
        cv2.putText(img,"This means an entire hand", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    '''

    cv2.namedWindow('Pet', cv2.WINDOW_NORMAL)
    tigger = cv2.imread('tigger.jpg')
    yuzu = cv2.imread('yuzu.jpg')
    glover = cv2.imread('glover.jpg')

    if count_defects > 3:
        str = "Tigger!"
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        cv2.imshow("Pet", tigger)
    elif count_defects == 2:
        str = "Yuzu :)"
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        cv2.imshow("Pet", yuzu)
    else:
        str = "Glover..."
        cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        cv2.imshow("Pet", glover)

    # show appropriate images in windows
    cv2.namedWindow('Gesture', cv2.WINDOW_NORMAL)
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.imshow('Contours', all_img)

    # Resize windows to display 
    cv2.resizeWindow('Pet', (400, 533))
    cv2.resizeWindow('Gesture', (946, 533))
    cv2.resizeWindow('Thresholded', (200, 200))
    cv2.resizeWindow('Contours', (400, 200))
    
    # Move windows to display
    cv2.moveWindow('Pet', 0, 0)
    cv2.moveWindow('Gesture', 400, 0)
    cv2.moveWindow('Thresholded', 400, 533)
    cv2.moveWindow('Contours', 600, 533)

    k = cv2.waitKey(10)
    # Press esc to stop screen
    if k == 27:
        break