if __name__ == "__main__":
    import sys
    import cv2
    import numpy as np
    
    image = cv2.imread("blank_keyboard.png", cv2.IMREAD_COLOR)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    
    # draw keyboard corners
    tl_corner = cv2.aruco.drawMarker(dictionary, id=1, sidePixels=100)
    tr_corner = cv2.aruco.drawMarker(dictionary, id=2, sidePixels=100)
    for r in range(100):
        for c in range(100):
            image[r, c] = tl_corner[r, c]
    for r in range(100):
        for c in range(100):
            image[r, -(100-c)] = tr_corner[r, c]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 3
    color = 0
    thickness = 4
    
    char_indicator = [252, 29, 0]
    
    current_char = 0
    char_keys = "qwertyuiopasdfghjklzxcvbnm"
    marker_length = 120
    textbox_height = 80
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if list(image[r, c]) == char_indicator:
                marker_img = cv2.aruco.drawMarker(dictionary, id=ord(char_keys[current_char]), sidePixels=marker_length)
                for r2 in range(marker_length):
                    for c2 in range(marker_length):
                        image[r + r2, c + c2] = marker_img[r2, c2]
                
                text = char_keys[current_char].upper()
                textsize, baseline = cv2.getTextSize(text, font, fontscale, thickness)
                x = int((marker_length - textsize[0])/2 + c)
                y = int((textbox_height + textsize[1])/2 + marker_length + r)
                new_image = cv2.putText(image, text, (x, y), font, fontscale, color, thickness)
                
                current_char += 1
        print(r)
    
    cv2.imwrite("keyboard_4X4_250.png", image)