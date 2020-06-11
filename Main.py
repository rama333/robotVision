import cv2

# np is an alias pointing to numpy library
import numpy as np


if __name__ == '__main__':
    def nothing(*arg):
        pass

cap = cv2.VideoCapture(0)

width  = cap.get(3) # float
height = cap.get(4) # float

print(width)
print(height)
cv2.namedWindow("settings")

cv2.createTrackbar('h1', 'settings', 25, 255, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 191, 255, nothing)
cv2.createTrackbar('h2', 'settings', 151, 255, nothing)
cv2.createTrackbar('s2', 'settings', 184, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
crange = [0, 0, 0, 0, 0, 0]

while (1):

    ret, frame = cap.read()


    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #img = cv2.medianBlur(frame, 5)
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray1,(5,5),cv2.BORDER_DEFAULT)

    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')


    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    thresh = cv2.inRange(hsv, h_min, h_max)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow('res', res)

    edges = cv2.Canny(frame, 100, 200)
    #gray = cv2.Canny(frame, 100, 200)

    #gray_blurred = cv2.blur(thresh, (3, 3))

    gray = cv2.GaussianBlur(thresh , (5, 5), 1)

    cv2.imshow('result', thresh)

    detected_circles = cv2.HoughCircles(gray,
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 90,
               param2 = 30, minRadius = 1, maxRadius = 140)

    if detected_circles is not None:

        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            cv2.circle(frame, (a, b), r, (0, 255, 0), 2)

            print((272*11)/r)
            #cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)

            # Display an original image
    cv2.imshow('Original', frame)

    #cv2.imshow('Edges', edges)

            # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()