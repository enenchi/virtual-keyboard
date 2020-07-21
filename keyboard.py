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
    
    TO_PRESSED_DELAY    = 0.15  # Time required in the DOWN history to transition RELEASED->TO_PRESSED
    PRESSED_DELAY       = 0.45  # Time required in the DOWN history to transition TO_PRESSED->PRESSED
    COVERED_DELAY       = 0.60  # Time required in the DOWN history to transition PRESSED->COVERED
    
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
                    pressed_keys.add(id) # add to pressed
            elif old_state == Keyboard.PRESSED:
                if sample == Keyboard.UP:
                    self.states[id] = Keyboard.RELEASED
                    released_keys.add(id) # add to released
                elif diff > Keyboard.COVERED_DELAY: # if state is PRESSED, the history has to be all DOWN
                    self.states[id] = Keyboard.COVERED
                    released_keys.add(id) # add to released
            elif old_state == Keyboard.COVERED:
                if sample == Keyboard.UP:
                    self.states[id] = Keyboard.RELEASED
            
            # update history
            if sample != prev_sample:
                self.histories[id] = (sample, timestamp)
            
            # fill return values
            transitions[id] = self.states[id] - old_state
            #if transitions[id] != 0:
            #   print("State:", chr(id), "<-", state)
        
        return transitions, pressed_keys, released_keys
    
    def reset(self):
        """
        Sets all states to RELEASED and all histories to UP at the current timestamp.
        Returns keys that went from pressed to released.
        """
        pressed_keys = set()
        timestamp = time.time()
        for id in self.states:
            if self.states[id] == Keyboard.PRESSED:
                pressed_keys.add(id)
            self.states[id] = Keyboard.RELEASED
            self.histories[id] = (Keyboard.UP, timestamp)
        return pressed_keys

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
    # apply y shift
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

def transform_coord(pair, M):
    coord = [pair[0], pair[1], 1]
    scalar = M[2].dot(coord)
    return [M[0].dot(coord) / scalar, M[1].dot(coord) / scalar]

def transform_corners(corners, M):
    copy = corners.copy()
    for corner in copy:
        corner[0] = np.array([transform_coord(pair, M) for pair in corner[0]], np.float32)
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
        M, size = get_transformation(markers)
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

def to_keyvalue(id):
    if id >= ord("a") and id <= ord("z"):
        return chr(id)
    raise ValueError("%d should not have been found" % id)

if __name__ == "__main__":
    import sys
    
    reference_image = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)
    KEYBOARD_DIM = reference_image.shape[1], reference_image.shape[0]
    keyboard = Keyboard(reference_image)
    all_keys = set(keyboard.ids())
    
    from pynput.keyboard import Controller
    controller = Controller()
    
    scale = 3 / 4
    resolution = (1920, 1080) # (1280, 720)
    scaled_resolution = tuple(map(lambda x: int(x * scale), resolution))
    disp_resolution = (640, 480)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_MONOCHROME, 1)
    
    marker1_box = [[0, 0], [2, 2]]
    marker2_box = [[0, 0], [2, 2]]
    keyboard_box = [[0, 0], [2, 2]]
    
    def get_roi_boxes(image, margin=1):
        print("Finding ROIs")
        markers, corners, ids, _ = get_markers(image)
        box1, box2, box = None, None, None
        if 1 in markers:
            x, y, w, h = cv2.boundingRect(markers[1])
            dx1 = w * margin
            dy1 = h * margin
            box1 = [[int(x - dx1), int(y - dy1)], [int(x + w + dx1), int(y + h + dy1)]]
        if 2 in markers:
            x, y, w, h = cv2.boundingRect(markers[2])
            dx2 = w * margin
            dy2 = h * margin
            box2 = [[int(x - dx2), int(y - dy2)], [int(x + w + dx2), int(y + h + dy2)]]
        if 2 in markers and 1 in markers:
            M, size = get_transformation(markers)
            M_inv = cv2.invert(M)[1]
            corners = [[[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]]]
            corners = transform_corners(corners, M_inv)
            x, y, w, h = cv2.boundingRect(corners[0][0])
            dx = max(dx1, dx2)
            dy = max(dy1, dy2)
            box = [[int(x - dx), int(y - dy)], [int(x + w + dx), int(y + h + dy)]]
        return box1, box2, box
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, scaled_resolution)
            
            # test marker 1 and 2
            if marker1_box != None:
                markers, _, _, _ = get_markers(frame[marker1_box[0][1]:marker1_box[1][1], marker1_box[0][0]:marker1_box[1][0]])
            
            if 1 not in markers:
                marker1_box, marker2_box, keyboard_box = get_roi_boxes(frame)
            else:
                if marker2_box != None:
                    markers, _, _, _ = get_markers(frame[marker2_box[0][1]:marker2_box[1][1], marker2_box[0][0]:marker2_box[1][0]])
                if 2 not in markers:
                    marker1_box, marker2_box, keyboard_box = get_roi_boxes(frame)
            
            if marker1_box != None and marker2_box != None and keyboard_box != None:
                #frame = frame[keyboard_box[0][1]:keyboard_box[1][1], keyboard_box[0][0]:keyboard_box[1][0]]
                markers, corners, ids, _ = get_markers(frame)
                
                # print which ids are found and which are missing
                found = set(markers.keys())
                if all_keys == found:
                    print("Found all markers")
                else:
                    print("Missing", all_keys.difference(found))
                    print("With extra", found.difference(all_keys))
                
                # draw
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                frame = cv2.rectangle(frame, tuple(marker1_box[0]), tuple(marker1_box[1]), 0, 2)
                frame = cv2.rectangle(frame, tuple(marker2_box[0]), tuple(marker2_box[1]), 0, 2)
                frame = cv2.rectangle(frame, tuple(keyboard_box[0]), tuple(keyboard_box[1]), 0, 2)
            else:
                print("Could not find markers 1 and 2")
            
            # show frame with all detected markers
            frame = cv2.resize(frame, disp_resolution)
            cv2.imshow('frame', frame)
        
            # break if the user pressed q to continue with the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        else:
            print("Reading from camera failed. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    cv2.destroyAllWindows()
    try:
        total = 0
        num = 0
        while True:
            ret, frame = cap.read()
            start_time = time.time()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, scaled_resolution)
                markers = {}
                
                # test marker 1 and 2
                if marker1_box != None:
                    markers, _, _, _ = get_markers(frame[marker1_box[0][1]:marker1_box[1][1], marker1_box[0][0]:marker1_box[1][0]])
                
                if 1 not in markers:
                    marker1_box, marker2_box, keyboard_box = get_roi_boxes(frame)
                else:
                    if marker2_box != None:
                        markers, _, _, _ = get_markers(frame[marker2_box[0][1]:marker2_box[1][1], marker2_box[0][0]:marker2_box[1][0]])
                    if 2 not in markers:
                        marker1_box, marker2_box, keyboard_box = get_roi_boxes(frame)
                
                if marker1_box != None and marker2_box != None and keyboard_box != None:
                    frame = frame[keyboard_box[0][1]:keyboard_box[1][1], keyboard_box[0][0]:keyboard_box[1][0]]
                
                    markers, _, _, _ = get_markers(frame)
                    
                    transitions, pressed_keys, released_keys = process_markers(markers, keyboard)
                    
                    for key in transitions:
                        value = to_keyvalue(key)
                        if key in pressed_keys:
                            controller.press(value)
                            print("Pressed:", value)
                        elif key in released_keys:
                            controller.release(value)
                            print("Released:", value)
                else:
                    print("Keyboard not found")
                    pressed_keys = keyboard.reset()
                    for key in pressed_keys:
                        value = to_keyvalue(key)
                        controller.release(value)
                        print("Reset:", value)
                """
                frame = cv2.resize(frame, disp_resolution)
                cv2.imshow('video', frame)
                cv2.waitKey(1)
                """
            else:
                break
            #print("--- %s seconds ---" % (time.time() - start_time))
            total += (time.time() - start_time)
            num += 1
    except KeyboardInterrupt:
        pass
    finally:
        print(total / num)
        cap.release()
        cv2.destroyAllWindows()
