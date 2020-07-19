import numpy as np
import cv2

import math
import time

# ArUco marker and keyboard information
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
KEYBOARD_DIM = None
MARKER_DIM = 100

REFERENCE = "keyboard_ref_4X4_250.png"

def build_keyboard(markers):
    # map pixel row to ids in that row
    board = dict()
    for id in markers:
        row = markers[id][0][1]
        if row in board:
            board[row].append(id)
        else:
            board[row] = [id]
    return [set(board[key]) for key in sorted(board.keys())]

class Keyboard:
    RELEASED    = 0
    TO_PRESSED  = 1
    PRESSED     = 2
    COVERED     = 3
    
    UP      = True
    DOWN    = False
    
    TO_PRESSED_DELAY    = 0.25  # Time required in the DOWN history to transition RELEASED->TO_PRESSED
    PRESSED_DELAY       = 0.75  # Time required in the DOWN history to transition TO_PRESSED->PRESSED
    COVERED_DELAY       = 1     # Time required in the DOWN history to transition PRESSED->COVERED
    
    def __init__(self, ref_image):
        # get markers from reference image
        markers, _, _, _ = get_markers(ref_image)
        
        # build states
        self.states = dict()
        
        # build the keyboard
        self.board = build_keyboard(markers)
        
        # build positions
        self.positions = dict()
        ref_height, ref_width = ref_image.shape[:2]
        
        # build histories
        self.histories = dict()
        
        timestamp = time.time()
        for id in markers:
            # states
            self.states[id] = Keyboard.RELEASED
            # positions
            corner = markers[id][0]
            self.positions[id] = corner[0] / ref_width, corner[1] / ref_height
            # histories and primed
            self.histories[id] = (Keyboard.UP, timestamp)
    
    def update(self, samples):
        transitions = dict()
        pressed_keys = set()
        released_keys = set()
        
        timestamp = time.time()
        for id in self.states:
            # determine sample; if id was in samples, the key sample is released
            sample = id in samples
            
            prev_sample, prev_time = self.histories[id]
            diff = timestamp - prev_time
            
            # update state machine
            old_state = self.states[id]
            if old_state == Keyboard.RELEASED:
                if sample == Keyboard.DOWN and prev_sample == Keyboard.DOWN and diff > Keyboard.TO_PRESSED_DELAY:
                    self.states[id] = Keyboard.TO_PRESSED
                    # cover all keys below this one
                    for i in range(len(self.board)):
                        if id in self.board[i]:
                            for row in self.board[i+1:]:
                                for id2 in row:
                                    self.states[id2] = Keyboard.COVERED
                                    #print("Deprimed:", ids)
                            break
            elif old_state == Keyboard.TO_PRESSED:
                if sample == Keyboard.UP:
                    self.states[id] = Keyboard.RELEASED
                elif diff > Keyboard.PRESSED_DELAY:
                    self.states[id] = Keyboard.PRESSED
            elif old_state == Keyboard.PRESSED:
                if sample == Keyboard.UP:
                    self.states[id] = Keyboard.RELEASED
                elif diff > Keyboard.COVERED_DELAY: # if state is PRESSED, the history has to be all DOWN
                    self.states[id] = Keyboard.COVERED
            elif old_state == Keyboard.COVERED:
                if sample == Keyboard.UP:
                    self.states[id] = Keyboard.RELEASED
            
            # update history
            if sample != prev_sample:
                self.histories[id] = (sample, timestamp)
            
            # fill return values
            transitions[id] = self.states[id] - old_state
            if transitions[id] != 0:
                state = self.states[id]
                if state == Keyboard.RELEASED:
                    released_keys.add(id)
                elif state == Keyboard.TO_PRESSED:
                    pass
                elif state == Keyboard.PRESSED:
                    pressed_keys.add(id)
                elif state == Keyboard.COVERED:
                    released_keys.add(id)
                #print("State:", chr(id), "<-", state)
        
        return transitions, pressed_keys, released_keys

    def ids(self):
        return set(self.states.keys())
    
    def draw_info(self, image, info, font=cv2.FONT_HERSHEY_SIMPLEX, fontscale=0.5, color=255, thickness=2):
        """
        Draws the string version of the info values into the corresponding key position
        """
        height, width = image.shape[:2]
        for id in self.states:
            x, y = self.positions[id]
            cv2.putText(image, str(info[id]), (int(x*width), int(y*height)), font, fontscale, color, thickness)

def show_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_transformation(markers):
    """
    Returns the transformation needed to isolate the keyboard and a tuple
    of the resulting image size.
    """
    # get bounds of markers
    q1 = markers[1][1]
    q2 = markers[2][0]
    q3 = markers[2][3]
    q4 = markers[1][2]
    src_rect = np.array([q1, q2, q3, q4], np.float32)
    
    # get bounds of destination markers
    box_ratio = KEYBOARD_DIM[0] / MARKER_DIM
    box_h = math.hypot(q3[0] - q2[0], q3[1] - q2[1])
    box_w = box_ratio * box_h
    
    r1 = [0, 0]
    r2 = [box_w, 0]
    r3 = [box_w, box_h]
    r4 = [0, box_h]
    dest_rect = np.array([r1, r2, r3, r4], np.float32)
    
    # get expected height of keyboard + box height
    keyboardbox_ratio = (KEYBOARD_DIM[1] + MARKER_DIM)/ KEYBOARD_DIM[0]
    expected_h = keyboardbox_ratio * box_w
    
    # get perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_rect, dest_rect)
    # add y shift
    for j in range(3):
        M[1][j] += M[2][j] * -box_h
    
    return M, (math.ceil(box_w), math.ceil(expected_h - box_h))

def get_markers(image):
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, DICTIONARY, parameters=parameters)
    
    markers = dict()
    # check if ids is array or None
    if type(ids) == np.ndarray:
        for id, corner in zip(ids, corners):
            markers[id[0]] = corner[0]
    
    return markers, corners, ids, rejectedImgPoints

def resize_image(image, scale):
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)
    return cv2.resize(image, (width, height))

def resize_corners(corners, scale):
    copy = corners.copy()
    for corner in copy:
        for pair in corner[0]:
            pair[0] *= scale
            pair[1] *= scale
    return copy

def transform_corners(corners, M):
    copy = corners.copy()
    for corner in copy:
        for pair in corner[0]:
            coord = [pair[0], pair[1], 1]
            scalar = M[2].dot(coord)
            pair[0] = M[0].dot(coord) / scalar
            pair[1] = M[1].dot(coord) / scalar
    return copy

def extract_keyboard(image, scale):
    # grayscale the image
    if len(image.shape) < 3:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect markers with full-sized image
    markers, corners, ids, rejected = get_markers(gray)
    if type(ids) == type(None):
        # TODO: remove drawing
        for reject in rejected:
            pts = np.array(rejected[0], np.int32)
            cv2.polylines(gray, [pts], True, 255, 4)
        
        return gray, markers
    
    # get tilted small image
    if (1 in markers) and (2 in markers):
        # get transform for original image
        M, size = markers_12(markers)
        # apply scalling
        for j in range(3):
            M[2][j] /= scale
        size = (int(size[0] * scale), int(size[1] * scale))
        
        image = cv2.warpPerspective(gray, M, size)
        corners = transform_corners(corners, M)
        
        # update markers
        for id, corner in zip(ids, corners):
            markers[id[0]] = corner[0]
        # TODO: remove
        #image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    else:
        image = cv2.aruco.drawDetectedMarkers(gray, corners, ids)
    return image, markers

def process_markers(markers, keyboard):
    """
    Returns transitions and the typed keys
    """
    return keyboard.update(set(markers.keys()))

def process_frame(frame, markers, keyboard, scale, draw_state=False, draw_transition=False):
    image, markers = extract_keyboard(frame, scale)
    
    state_scale = image.shape[1] / KEYBOARD_DIM[0]
    if draw_state:
        states.draw_states(image, state_scale)
    if draw_transition:
        states.draw_transitions(image, transitions, state_scale)
    
    return process_markers(markers, keyboard)

"""
if __name__ == "__main__":
    import sys
    
    ref_image = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)
    markers, _, _, _ = get_markers(ref_image)
    board = build_keyboard(markers)
    for row in board:
        for i in range(len(row)):
            row[i] = chr(row[i])
    print(board)
    
    sys.exit(1)
"""

def to_keyvalue(id):
    if id >= ord("a") and id <= ord("z"):
        return chr(id)
    raise ValueError("%d should not have been found" % id)

if __name__ == "__main__":
    import sys
    
    scale = 1
    
    reference_image = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)
    KEYBOARD_DIM = reference_image.shape[1], reference_image.shape[0]
    keyboard = Keyboard(reference_image)
    
    if len(sys.argv) == 1:
        from pynput.keyboard import Controller
        controller = Controller()
        
        all_keys = set(keyboard.ids())
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)
        #cap.set(3, 1280)
        #cap.set(4, 720)
        
        #disp_scale = 1 / 4
        #disp_size = (int(KEYBOARD_DIM[0] * disp_scale), int(KEYBOARD_DIM[1] * disp_scale))
        scale = 3 / 4
        
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = frame.shape[:2]
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                markers, corners, ids, _ = get_markers(frame)
                
                # print which ids are found and which are missing
                found = set(markers.keys())
                if all_keys == found:
                    print("Found all markers")
                else:
                    print("Missing", all_keys.difference(found))
                    print("With extra", found.difference(all_keys))
                
                # show frame with all detected markers
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.imshow('frame', frame)
                
                # break if the user pressed q to continue with the program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Reading from camera failed. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        
        cv2.destroyAllWindows()
        try:
            while True:
                ret, frame = cap.read()
                start_time = time.time()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    height, width = frame.shape[:2]
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                    
                    markers, _, _, _ = get_markers(frame)
                    #print("# markers:", len(markers))
                    transitions, pressed_keys, released_keys = process_markers(markers, keyboard)
                    
                    for key in transitions:
                        if key in pressed_keys:
                            value = to_keyvalue(key)
                            controller.press(value)
                            print("Pressed:", value)
                        elif key in released_keys:
                            value = to_keyvalue(key)
                            controller.release(value)
                            print("Released:", value)
                    
                    """
                    frame = cv2.resize(frame, disp_size)
                    cv2.imshow('video', frame)
                    cv2.waitKey(1)
                    """
                else:
                    break
                print("--- %s seconds ---" % (time.time() - start_time))
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
        sys.exit(0)
    
    if len(sys.argv) != 3:
        print("Incorrect argument amount [-i|-m filename]")
        sys.exit(1)
    
    if sys.argv[1] == "-i":
        start_time = time.time()
        
        frame = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
        
        image, typed_keys = process_frame(frame, states, keys, scale)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        cv2.imwrite("out.jpg", image)
        show_image("final image", image)
    elif sys.argv[1] == "-m":
        cap = cv2.VideoCapture(sys.argv[2])
        
        frames = []
        
        start_time = time.time()
        while(cap.isOpened()):
            
            ret, frame = cap.read()
            time.sleep(1/30)
            
            if ret:
                image, typed_keys = process_frame(frame, states, keys, scale)
                
                
                chars = []
                if typed_keys:
                    for key in typed_keys:
                        chars.append(chr(key))
                    print(chars)
        
                #cv2.putText(image, str(chars) + str(time.time() - start_time), (0, image.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
                
                frames.append(image)    
                
                if len(frames) >= 100:
                    break
            else:
                break
        print("--- avg %s seconds per frame---" % ((time.time() - start_time) / len(frames)))
        
        sys.exit(1)
        
        import os
        import shutil
        dirname = "%s_output" % sys.argv[2][:sys.argv[2].rfind(".")]
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)
        
        height, width = frames[0].shape[:2]
        height += height % 2
        width += width % 2
        for index, frame in zip(range(len(frames)), frames):
            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(os.path.join(dirname, "%05d.png" % index), frame)
        
        os.system("ffmpeg -framerate 30 -pattern_type glob -i '%s/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4" % dirname)
        
        cap.release()
        cv2.destroyAllWindows()
    