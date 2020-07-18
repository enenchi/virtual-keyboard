import numpy as np
import cv2

import math
import time

# ArUco marker and keyboard information
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
KEYBOARD_DIM = (3273, 1228)
MARKER_DIM = 100

# state information
class States:
    _REF = "keyboard_ref_4X4_250.png"
    RELEASED    = 1
    PRESSED     = 0
    RELEASED_TO_PRESSED = RELEASED - PRESSED
    PRESSED_TO_RELEASED = PRESSED - RELEASED
    
    def to_text(id):
        if id >= ord("a") and id <= ord("z"):
            return chr(id - ord("a") + ord("A"))
        raise ValueError("%d should not have been found" % id)
    
    def __init__(self):
        # read reference image
        reference = cv2.imread(States._REF, cv2.IMREAD_GRAYSCALE)
        markers, _, _, _ = get_markers(reference)
        
        # initialize states and positions
        self.states = dict()
        self.positions = dict()
        for id in markers:
            self.states[id] = States.RELEASED
            self.positions[id] = int(markers[id][0][0]), int(markers[id][0][1])
    
    def update(self, ids):
        """
        Updates the states. Returns the state transitions
        """
        transitions = dict()
        for id in self.states:
            old_state = self.states[id]
            new_state = States.RELEASED if id in ids else States.PRESSED
            self.states[id] = new_state
            transitions[id] = old_state - new_state
            #if old_state - new_state != 0:
                #print("State: %s <- %d" % (chr(id), new_state))
        return transitions
    
    def draw_states(self, image, scale, font=cv2.FONT_HERSHEY_SIMPLEX, fontscale=0.5, color=255, thickness=2):
        for id in self.states:
            x, y = self.positions[id]
            x = int(x * scale)
            y = int(y * scale)
            cv2.putText(image, str(self.states[id]), (x, y), font, fontscale, color, thickness)
    
    def draw_transitions(self, image, transitions, scale, font=cv2.FONT_HERSHEY_SIMPLEX, fontscale=0.5, color=255, thickness=2):
        for id in transitions:
            text = States.to_text(id)
            if transitions[id] == States.PRESSED_TO_RELEASED:
                text = "-" + text
            elif transitions[id] == States.RELEASED_TO_PRESSED:
                text = "+" + text
            else:
                text = ""
            x, y = self.positions[id]
            x = int(x * scale)
            y = int(y * scale)
            cv2.putText(image, str(transitions[id]), (x, y), font, fontscale, color, thickness)
    
    def __iter__(self):
        return iter(self.states)
    
    def __getitem__(self, item):
        return self.states[item]
    
    def ids(self):
        return list(self.states.keys())

class Keyboard:
    DEPRIME_TIME = 0.75
    MIN_DELAY = 1
    MAX_DELAY = 2
    
    ROWS_CH = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    ROWS_ID = None
    
    def __init__(self, states):
        self.histories = dict()
        self.primed = dict()
        timestamp = time.time()
        for id in states:
            self.histories[id] = [(states[id], timestamp)]
            self.primed[id] = False
        if Keyboard.ROWS_ID == None:
            Keyboard.ROWS_ID = []
            for row in Keyboard.ROWS_CH:
                Keyboard.ROWS_ID.append(set([ord(c) for c in row]))
    
    def update(self, states):
        typed_keys = []
        timestamp = time.time()
        for id in states:
            state = states[id]
            prev, prev_time = self.histories[id][-1]
            if state == States.PRESSED and prev == States.PRESSED:
                diff = timestamp - prev_time
                # de-prime rows under this id
                if diff > Keyboard.DEPRIME_TIME:
                    for i in range(len(Keyboard.ROWS_ID)):
                        if id in Keyboard.ROWS_ID[i]:
                            for i in range(i + 1, len(Keyboard.ROWS_ID)):
                                for id in Keyboard.ROWS_ID[i]:
                                    self.primed[id] = False
                                    #print("De-primed %s" % chr(id))
            if prev != state:
                self.histories[id].append((state, timestamp))
                if state == States.PRESSED:
                    self.primed[id] = True
                elif state == States.RELEASED:
                    if self.primed[id]:
                        diff = timestamp - prev_time
                        if diff > Keyboard.MIN_DELAY and diff < Keyboard.MAX_DELAY:
                            typed_keys.append(id)
                            #print(chr(id), timestamp - prev_time, self.histories[id])
                    self.primed[id] = False
                if len(self.histories[id]) > 2:
                    self.histories[id] = self.histories[id][-2:]
        return typed_keys

def show_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def markers_12(markers):
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

def process_markers(markers, states, keys):
    """
    Returns transitions and the typed keys
    """
    transitions = states.update(markers.keys())
    typed_keys = keys.update(states)
    
    return transitions, typed_keys

def process_frame(frame, markers, states, keys, scale, draw_state=False, draw_transition=False):
    image, markers = extract_keyboard(frame, scale)
    
    state_scale = image.shape[1] / KEYBOARD_DIM[0]
    if draw_state:
        states.draw_states(image, state_scale)
    if draw_transition:
        states.draw_transitions(image, transitions, state_scale)
    
    return process_markers(markers, states, keys)

if __name__ == "__main__":
    import sys
    
    scale = 1
    states = States()
    keys = Keyboard(states)
    
    if len(sys.argv) == 1:
        from pynput.keyboard import Key, Controller
        controller = Controller()
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        #cap.set(3, 1280)
        cap.set(4, 1080)
        #cap.set(4, 720)
        
        disp_scale = 1 / 4
        size = (int(KEYBOARD_DIM[0] * disp_scale), int(KEYBOARD_DIM[1] * disp_scale))
        ret, frame = cap.read()
        markers, _, _, _ = get_markers(frame)
        while True:
            ret, frame = cap.read()
            if ret:
                markers, corners, ids, _ = get_markers(frame)
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                ids = set(states.ids())
                found = set(markers.keys())
                if ids == found:
                    print("Found all markers")
                else:
                    print("Missing", ids.difference(found))
                    print("With extra", found.difference(ids))
                frame = cv2.resize(frame, size)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        try:
            while True:
                ret, frame = cap.read()
                #start_time = time.time()
                if ret:
                    markers, _, _, _ = get_markers(frame)
                    #print("# markers:", len(markers))
                    transitions, typed_keys = process_markers(markers, states, keys)
                    
                    if typed_keys:
                        for key in typed_keys:
                            print(chr(key))
                            controller.type(chr(key))
                        #print(chars)
                    
                    """
                    image = cv2.resize(image, size)
                    cv2.imshow('video', image)
                    cv2.waitKey(1)
                    """
                else:
                    break
                #print("--- %s seconds ---" % (time.time() - start_time))
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
    