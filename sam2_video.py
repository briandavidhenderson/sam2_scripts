# Prepare imports
import torch
import cv2
import numpy as np
import os
import sys
import time
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import h5py  # For saving data
import json  # For saving prompts
# <<< Added for performance and memory management >>>
from torch.cuda.amp import autocast as autocast # For mixed precision - Use torch.amp.autocast('cuda', enabled=...) in newer PyTorch
import gc  # For garbage collection

# --- Configuration ---
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Relative to sam2 package

INPUT_VIDEO_MP4 = r""  # <<< Path to input video
TEMP_FRAME_DIR = "temp_frames" # Changed temp dir name
OUTPUT_DIR = "" # Changed output dir name
OUTPUT_VIDEO_NAME = "_video_.mp4"
OUTPUT_HDF5_FILE = "_video_.hdf5"
OUTPUT_PROMPTS_JSON = "_video_.json" # Renamed

# --- >>> Resizing Configuration <<< ---
# Adjust resolution based on GPU memory. Lower if OOM errors occur.
TARGET_WIDTH = 3000 # Example: 1280 might work with chunking, adjust if needed
TARGET_HEIGHT = None # Set to None to scale based on width

# --- Memory Management Configuration ---
CHUNK_SIZE = 50 # Process video in chunks of this many frames. Adjust based on memory.
ENABLE_CHUNKING = True # <<< MUST BE TRUE for this modified logic >>>

# --- GUI Constants ---
BUTTON_HEIGHT = 30
BUTTON_PADDING = 5
TRACKBAR_HEIGHT = 50 # Space for frame slider
HEADER_HEIGHT = BUTTON_HEIGHT * 2 + BUTTON_PADDING * 3 + TRACKBAR_HEIGHT # Updated by define_buttons

# --- Global variables for GUI state ---
drawing_data = {
    # --- NEW: Store prompts per frame index ---
    "interactions_by_frame": {}, # {frame_idx: {obj_id: {"class": str, "points": list, "labels": list, "box": list}}}
    # ---
    "next_global_id": 1,
    "class_names": ["Object", "Cell Type A", "Cell Type B", "Debris", "Other"],  # <<< Define your classes here
    "current_class_index": 0,
    "current_mode": 'p', # p=pos, n=neg, b=box
    "is_adding_to_new_object": True,
    "current_interaction_obj_id": None, # Which object ID subsequent clicks belong to (until 'O' is pressed)
    "drawing_box": False,
    "box_start_pt_img": None,  # Store box start in IMAGE coordinates
    "temp_box_end_pt_win": None, # Store temp box end in WINDOW coordinates
    "window_name": "Select Prompts - Scroll=Zoom, M-Click+Drag=Pan, Slider=Frame, Keys: [P]os [N]eg [B]ox [O]bject [R]eset [F]inish",
    "buttons": {},
    "exit_flag": False,
    # --- Zoom/Pan State ---
    "zoom_level": 1.0,
    "view_offset_x": 0.0, # Top-left corner of the view in IMAGE coordinates
    "view_offset_y": 0.0,
    "panning": False,
    "pan_start_x": 0,
    "pan_start_y": 0,
    # --- Frame/Window Info ---
    "base_image_width": 0,
    "base_image_height": 0,
    "window_width": 0,
    "window_height": 0,
    "image_area_height": 0, # Height of the interactive image area below header
    # --- NEW: Multi-frame GUI State ---
    "current_frame_index": 0,
    "total_frames_for_gui": 0,
    "temp_frame_dir_gui": "", # Path to pre-extracted frames for GUI
    "base_frames_gui": {} # Cache loaded frames for GUI {frame_idx: numpy_array} - Limited cache
}

# --- Helper Functions ---

def get_color(obj_id):
    """Gets a distinct color for an object ID using tab10 colormap."""
    cmap = plt.get_cmap("tab10")
    color_rgb = cmap(obj_id % 10)[:3]
    color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])
    return color_bgr

# --- Coordinate Transformation Helpers (Unchanged) ---
def window_to_image_coords(win_x, win_y):
    """Converts window coordinates (within image area) to full image coordinates."""
    global drawing_data
    img_area_y = win_y - HEADER_HEIGHT # Adjust for potentially larger header
    img_x = drawing_data["view_offset_x"] + win_x / drawing_data["zoom_level"]
    img_y = drawing_data["view_offset_y"] + img_area_y / drawing_data["zoom_level"]
    return int(round(img_x)), int(round(img_y))

def image_to_window_coords(img_x, img_y):
    """Converts full image coordinates to window coordinates (within image area)."""
    global drawing_data
    win_x_unoffset = (img_x - drawing_data["view_offset_x"]) * drawing_data["zoom_level"]
    win_y_unoffset = (img_y - drawing_data["view_offset_y"]) * drawing_data["zoom_level"]
    win_y = win_y_unoffset + HEADER_HEIGHT # Adjust for potentially larger header
    return int(round(win_x_unoffset)), int(round(win_y))

# --- MODIFIED Drawing Functions ---
def draw_prompts_for_current_frame(frame_display_slice):
    """Draws existing points/boxes FOR THE CURRENT FRAME onto the display slice, considering zoom/pan."""
    global drawing_data
    current_fidx = drawing_data["current_frame_index"]
    # Get prompts specifically defined for the current frame
    interactions = drawing_data["interactions_by_frame"].get(current_fidx, {})

    win_slice_h, win_slice_w = frame_display_slice.shape[:2]

    for obj_id, data in interactions.items():
        color = get_color(obj_id)

        # Draw points if they exist for this object ON THIS FRAME
        if 'points' in data and data['points'] is not None:
            for i, pt_img in enumerate(data['points']): # pt_img is in full image coords
                if 'labels' in data and data['labels'] is not None and i < len(data['labels']):
                    label = data['labels'][i]
                    marker = cv2.MARKER_STAR if label == 1 else cv2.MARKER_CROSS
                    draw_win_x, draw_win_y_unoffset = image_to_window_coords(pt_img[0], pt_img[1])
                    draw_win_y = draw_win_y_unoffset - HEADER_HEIGHT # Adjust for slice offset

                    if 0 <= draw_win_x < win_slice_w and 0 <= draw_win_y < win_slice_h:
                        cv2.drawMarker(frame_display_slice, (draw_win_x, draw_win_y), color, markerType=marker, markerSize=15, thickness=2)

        # Draw box if it exists for this object ON THIS FRAME
        if data.get('box') is not None:
            x1_img, y1_img, x2_img, y2_img = data['box'] # Box is in full image coords
            win_x1, win_y1_unoffset = image_to_window_coords(x1_img, y1_img)
            win_x2, win_y2_unoffset = image_to_window_coords(x2_img, y2_img)
            win_y1 = win_y1_unoffset - HEADER_HEIGHT # Adjust for slice offset
            win_y2 = win_y2_unoffset - HEADER_HEIGHT

            draw_x1 = max(0, win_x1)
            draw_y1 = max(0, win_y1)
            draw_x2 = min(win_slice_w - 1, win_x2)
            draw_y2 = min(win_slice_h - 1, win_y2)

            if draw_x1 < draw_x2 and draw_y1 < draw_y2:
                cv2.rectangle(frame_display_slice, (draw_x1, draw_y1), (draw_x2, draw_y2), color, 2)

    # Draw temporary box being drawn by user (unchanged logic, applies to current frame)
    if drawing_data["drawing_box"] and drawing_data["box_start_pt_img"] and drawing_data["temp_box_end_pt_win"]:
        current_id = drawing_data["current_interaction_obj_id"]
        if current_id is not None:
            start_x_img, start_y_img = drawing_data["box_start_pt_img"]
            start_x_win, start_y_win_unoffset = image_to_window_coords(start_x_img, start_y_img)
            start_y_win = start_y_win_unoffset - HEADER_HEIGHT

            end_x_win, end_y_win_unoffset = drawing_data["temp_box_end_pt_win"]
            end_y_win = end_y_win_unoffset - HEADER_HEIGHT

            draw_x1 = max(0, min(start_x_win, end_x_win))
            draw_y1 = max(0, min(start_y_win, end_y_win))
            draw_x2 = min(win_slice_w - 1, max(start_x_win, end_x_win))
            draw_y2 = min(win_slice_h - 1, max(start_y_win, end_y_win))

            if draw_x1 < draw_x2 and draw_y1 < draw_y2:
                cv2.rectangle(frame_display_slice, (draw_x1, draw_y1),
                              (draw_x2, draw_y2), get_color(current_id), 1, cv2.LINE_AA)


def define_buttons(frame_width):
    """Calculates and stores button regions, accounting for trackbar space."""
    global drawing_data, HEADER_HEIGHT
    buttons = {}
    # Start buttons BELOW the trackbar space
    y_pos = BUTTON_PADDING + TRACKBAR_HEIGHT
    x_pos = BUTTON_PADDING
    num_classes = len(drawing_data["class_names"])
    total_class_padding = BUTTON_PADDING * (num_classes + 1)
    available_width = frame_width - total_class_padding
    max_width_per_button = available_width // num_classes if num_classes > 0 else frame_width

    # Class Buttons
    for i, name in enumerate(drawing_data["class_names"]):
        display_name = name[:12] + "..." if len(name) > 14 else name
        (w, h), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        button_w = min(max(w + 20, 60), max_width_per_button)

        if x_pos + button_w + BUTTON_PADDING > frame_width and i > 0:
            y_pos += BUTTON_HEIGHT + BUTTON_PADDING
            x_pos = BUTTON_PADDING
            remaining_buttons = num_classes - i
            max_width_per_button = (frame_width - BUTTON_PADDING * (remaining_buttons + 1)) // remaining_buttons if remaining_buttons > 0 else frame_width
            button_w = min(max(w + 20, 60), max_width_per_button) # Recalculate width

        buttons[name] = (x_pos, y_pos, button_w, BUTTON_HEIGHT, 'set_class', i)
        x_pos += button_w + BUTTON_PADDING

    # New row
    y_pos += BUTTON_HEIGHT + BUTTON_PADDING
    x_pos = BUTTON_PADDING

    # Action Buttons (Unchanged logic)
    action_buttons = {
        '[P]os Point': ('set_mode', 'p'),
        '[N]eg Point': ('set_mode', 'n'),
        '[B]ox': ('set_mode', 'b'),
        '[O]bject': ('new_object', None),
        '[R]eset': ('reset', None), # Reset now resets ALL frames
        '[F]inish': ('finish', None)
    }
    num_action_buttons = len(action_buttons)
    action_button_padding = BUTTON_PADDING * (num_action_buttons + 1)
    action_available_width = frame_width - action_button_padding
    action_button_w = action_available_width // num_action_buttons if num_action_buttons > 0 else frame_width
    action_button_w = max(action_button_w, 90)

    current_row_width = 0
    for name, (action_type, action_value) in action_buttons.items():
        if x_pos + action_button_w + BUTTON_PADDING > frame_width and x_pos != BUTTON_PADDING:
            y_pos += BUTTON_HEIGHT + BUTTON_PADDING
            x_pos = BUTTON_PADDING
            remaining_action_buttons = num_action_buttons - list(action_buttons.keys()).index(name)
            action_button_w = (frame_width - BUTTON_PADDING * (remaining_action_buttons + 1 )) \
                                // remaining_action_buttons if remaining_action_buttons > 0 else frame_width
            action_button_w = max(action_button_w, 90)

        buttons[name] = (x_pos, y_pos, action_button_w, BUTTON_HEIGHT, action_type, action_value)
        x_pos += action_button_w + BUTTON_PADDING

    drawing_data["buttons"] = buttons
    # Header height includes trackbar, buttons, and padding
    HEADER_HEIGHT = y_pos + BUTTON_HEIGHT + BUTTON_PADDING


def draw_buttons_and_info(frame):
    """Draws buttons, trackbar, and frame info onto the frame header area."""
    global drawing_data
    win_h, win_w = frame.shape[:2]

    # Draw header background
    cv2.rectangle(frame, (0, 0), (win_w, HEADER_HEIGHT), (40, 40, 40), cv2.FILLED)

    # --- Draw Trackbar Area (Placeholder - actual trackbar is OS element) ---
    trackbar_y_start = BUTTON_PADDING
    trackbar_x_start = BUTTON_PADDING
    trackbar_width = win_w - 2 * BUTTON_PADDING
    cv2.rectangle(frame, (trackbar_x_start, trackbar_y_start),
                  (trackbar_x_start + trackbar_width, trackbar_y_start + TRACKBAR_HEIGHT - BUTTON_PADDING),
                  (60, 60, 60), cv2.FILLED)
    cv2.putText(frame, f"Frame: {drawing_data['current_frame_index']} / {drawing_data['total_frames_for_gui'] - 1}",
                (trackbar_x_start + 5, trackbar_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    active_obj_text = f"Active Obj: {drawing_data['current_interaction_obj_id']}" if drawing_data['current_interaction_obj_id'] else "Active Obj: None (New)"
    (tw,th),_ = cv2.getTextSize(active_obj_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, active_obj_text,
                (win_w - tw - 15 , trackbar_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


    # --- Draw Buttons (below trackbar) ---
    for label, (x, y, w, h, action_type, action_value) in drawing_data["buttons"].items():
        is_active = False
        if action_type == 'set_class' and action_value == drawing_data["current_class_index"]:
            is_active = True
        elif action_type == 'set_mode' and action_value == drawing_data["current_mode"]:
            is_active = True

        bg_color = (100, 100, 100) if is_active else (70, 70, 70)
        text_color = (255, 255, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)

        display_label = label
        if action_type == 'set_class':
             if action_value < len(drawing_data['class_names']):
                 class_name = drawing_data['class_names'][action_value]
                 display_label = class_name
             else:
                 display_label = "Error"

        (text_w, text_h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        cv2.putText(frame, display_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


# --- GUI Action Handlers (Minor Adjustments) ---
def handle_set_class(class_index):
    global drawing_data
    if class_index != drawing_data["current_class_index"]:
        drawing_data["current_class_index"] = class_index
        # When changing class, assume user wants to start a NEW object of that class
        drawing_data["is_adding_to_new_object"] = True
        drawing_data["current_interaction_obj_id"] = None # Clear active object
        drawing_data["drawing_box"] = False # Cancel any box drawing
        drawing_data["box_start_pt_img"] = None
        drawing_data["temp_box_end_pt_win"] = None
        print(f"Selected Class: {drawing_data['class_names'][class_index]} (Ready for new object on next click)")

def handle_set_mode(mode):
    global drawing_data
    if mode != drawing_data["current_mode"]:
        drawing_data["current_mode"] = mode
        print(f"Mode set to: {mode}")
        if mode != 'b' and drawing_data["drawing_box"]: # Cancel box drawing if switching mode
            drawing_data["drawing_box"] = False
            drawing_data["box_start_pt_img"] = None
            drawing_data["temp_box_end_pt_win"] = None

def handle_new_object():
    global drawing_data
    drawing_data["is_adding_to_new_object"] = True
    drawing_data["current_interaction_obj_id"] = None # Clear active object ID
    drawing_data["drawing_box"] = False # Cancel box drawing
    drawing_data["box_start_pt_img"] = None
    drawing_data["temp_box_end_pt_win"] = None
    print(f"Ready to start NEW object (Class: {drawing_data['class_names'][drawing_data['current_class_index']]}) on next click/drag.")

def handle_reset():
    global drawing_data
    # Reset EVERYTHING - prompts on all frames, IDs, modes, view
    drawing_data["interactions_by_frame"] = {}
    drawing_data["next_global_id"] = 1
    drawing_data["is_adding_to_new_object"] = True
    drawing_data["current_interaction_obj_id"] = None
    drawing_data["current_class_index"] = 0
    drawing_data["current_mode"] = 'p'
    drawing_data["drawing_box"] = False
    drawing_data["box_start_pt_img"] = None
    drawing_data["temp_box_end_pt_win"] = None
    drawing_data["zoom_level"] = 1.0
    drawing_data["view_offset_x"] = 0.0
    drawing_data["view_offset_y"] = 0.0
    # Resetting frame index requires updating the trackbar too
    drawing_data["current_frame_index"] = 0
    cv2.setTrackbarPos('Frame', drawing_data["window_name"], 0)
    print("Reset ALL prompts on ALL frames, IDs, and view.")

def handle_finish():
    global drawing_data
    drawing_data["exit_flag"] = True

# --- NEW: Trackbar Callback ---
def on_trackbar(val):
    """Callback function when the trackbar is moved."""
    global drawing_data
    if val != drawing_data["current_frame_index"]:
        drawing_data["current_frame_index"] = val
        # print(f"Navigated to frame: {val}") # Optional debug
        # When navigating, cancel any ongoing box drawing on the *previous* frame
        if drawing_data["drawing_box"]:
            drawing_data["drawing_box"] = False
            drawing_data["box_start_pt_img"] = None
            drawing_data["temp_box_end_pt_win"] = None
            print("Cancelled box drawing due to frame navigation.")

# --- MODIFIED: Mouse Callback ---
def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks/movement for buttons, drawing, panning, zooming on the CURRENT frame."""
    global drawing_data

    # --- Button Click Handling (Only on Left Button Down) ---
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within the *button* area (below trackbar)
        if y >= TRACKBAR_HEIGHT and y < HEADER_HEIGHT:
            button_clicked = False
            for label, (bx, by, bw, bh, action_type, action_value) in drawing_data["buttons"].items():
                if bx <= x < bx + bw and by <= y < by + bh:
                    print(f"Button '{label}' clicked.")
                    button_clicked = True
                    if action_type == 'set_class': handle_set_class(action_value)
                    elif action_type == 'set_mode': handle_set_mode(action_value)
                    elif action_type == 'new_object': handle_new_object()
                    elif action_type == 'reset': handle_reset()
                    elif action_type == 'finish': handle_finish()
                    break
            if button_clicked:
                return # Don't process drawing if a button was clicked

    # --- Zoom Handling (Mouse Wheel) ---
    if event == cv2.EVENT_MOUSEWHEEL:
        if y >= HEADER_HEIGHT: # Ensure mouse is within the image area
            img_x_before, img_y_before = window_to_image_coords(x, y)
            delta = flags / 120
            zoom_factor = 1.1 if delta > 0 else (1/1.1)
            new_zoom_level = drawing_data["zoom_level"] * zoom_factor
            min_zoom = 0.5
            max_zoom = 15.0
            new_zoom_level = max(min_zoom, min(max_zoom, new_zoom_level))

            if new_zoom_level != drawing_data["zoom_level"]:
                drawing_data["zoom_level"] = new_zoom_level
                img_x_after_relative = x / drawing_data["zoom_level"]
                img_y_after_relative = (y - HEADER_HEIGHT) / drawing_data["zoom_level"]
                drawing_data["view_offset_x"] = img_x_before - img_x_after_relative
                drawing_data["view_offset_y"] = img_y_before - img_y_after_relative

                # Clamp offsets (using potentially updated image_area_height)
                current_image_area_height = drawing_data["window_height"] - HEADER_HEIGHT
                max_offset_x = max(0.0, drawing_data["base_image_width"] - (drawing_data["window_width"] / drawing_data["zoom_level"]))
                max_offset_y = max(0.0, drawing_data["base_image_height"] - (current_image_area_height / drawing_data["zoom_level"]))
                drawing_data["view_offset_x"] = max(0.0, min(drawing_data["view_offset_x"], max_offset_x))
                drawing_data["view_offset_y"] = max(0.0, min(drawing_data["view_offset_y"], max_offset_y))
            return

    # --- Panning Handling (Middle Mouse Button) ---
    if event == cv2.EVENT_MBUTTONDOWN:
        if y >= HEADER_HEIGHT:
            drawing_data["panning"] = True
            drawing_data["pan_start_x"] = x
            drawing_data["pan_start_y"] = y
            return
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing_data["panning"]:
            dx = x - drawing_data["pan_start_x"]
            dy = y - drawing_data["pan_start_y"]
            drawing_data["view_offset_x"] -= dx / drawing_data["zoom_level"]
            drawing_data["view_offset_y"] -= dy / drawing_data["zoom_level"]

            # Clamp offsets (using potentially updated image_area_height)
            current_image_area_height = drawing_data["window_height"] - HEADER_HEIGHT
            max_offset_x = max(0.0, drawing_data["base_image_width"] - (drawing_data["window_width"] / drawing_data["zoom_level"]))
            max_offset_y = max(0.0, drawing_data["base_image_height"] - (current_image_area_height / drawing_data["zoom_level"]))
            drawing_data["view_offset_x"] = max(0.0, min(drawing_data["view_offset_x"], max_offset_x))
            drawing_data["view_offset_y"] = max(0.0, min(drawing_data["view_offset_y"], max_offset_y))

            drawing_data["pan_start_x"] = x
            drawing_data["pan_start_y"] = y
            return
    if event == cv2.EVENT_MBUTTONUP:
        if drawing_data["panning"]:
            drawing_data["panning"] = False
            return

    # --- Drawing Interaction Handling (Left Button - Click/Drag in Image Area) ---
    if y < HEADER_HEIGHT or drawing_data["panning"]: # Ignore clicks in header or during panning
        return

    image_x, image_y = window_to_image_coords(x, y)
    mode = drawing_data["current_mode"]
    current_fidx = drawing_data["current_frame_index"] # Get current frame index

    # --- Get or create the interaction dictionary for the current frame ---
    if current_fidx not in drawing_data["interactions_by_frame"]:
        drawing_data["interactions_by_frame"][current_fidx] = {}
    frame_interactions = drawing_data["interactions_by_frame"][current_fidx]

    # --- Logic for Handling Clicks ---
    if event == cv2.EVENT_LBUTTONDOWN:
        obj_id_to_modify = None
        if drawing_data["is_adding_to_new_object"]:
            # --- Create New Global Object ID ---
            obj_id = drawing_data["next_global_id"]
            drawing_data["next_global_id"] += 1
            drawing_data["current_interaction_obj_id"] = obj_id # Set this as the active object
            drawing_data["is_adding_to_new_object"] = False # Now adding prompts to *this* specific object ID
            obj_id_to_modify = obj_id
            class_name = drawing_data["class_names"][drawing_data["current_class_index"]]
            print(f"--- Started Object {obj_id} (Class: {class_name}) on Frame {current_fidx} ---")
            # Ensure object entry exists for this frame even if prompt fails
            if obj_id not in frame_interactions:
                 frame_interactions[obj_id] = {'class': class_name, 'points': [], 'labels': [], 'box': None}
        else:
            # --- Add to Existing Object ID ---
            obj_id_to_modify = drawing_data["current_interaction_obj_id"]
            if obj_id_to_modify is None:
                 print("Error: No object selected to add prompts to. Click '[O]bject' first or select a class.")
                 return
            # Ensure object entry exists for this frame if adding refinement
            if obj_id_to_modify not in frame_interactions:
                 # Need to find the class associated with this ID from *any* frame
                 obj_class = 'Unknown'
                 for f_idx, f_data in drawing_data["interactions_by_frame"].items():
                     if obj_id_to_modify in f_data:
                         obj_class = f_data[obj_id_to_modify].get('class', 'Unknown')
                         break
                 frame_interactions[obj_id_to_modify] = {'class': obj_class, 'points': [], 'labels': [], 'box': None}
                 print(f"Adding refinement prompts for existing Object {obj_id_to_modify} (Class: {obj_class}) on Frame {current_fidx}")


        # Get the data dict for the object *on this frame*
        obj_data_on_frame = frame_interactions[obj_id_to_modify]

        # --- Add Prompt Based on Mode ---
        if mode in ['p', 'n']:
            label = 1 if mode == 'p' else 0
            if obj_data_on_frame.get('points') is None: # Initialize if first point for this obj on this frame
                 obj_data_on_frame['points'] = []
                 obj_data_on_frame['labels'] = []
            obj_data_on_frame['points'].append([image_x, image_y])
            obj_data_on_frame['labels'].append(label)
            pt_type = "positive" if label == 1 else "negative"
            print(f"  Added {pt_type} point win({x},{y}) -> img({image_x},{image_y}) for object {obj_id_to_modify} on frame {current_fidx}")

        elif mode == 'b':
            # Start drawing box (overwrites existing box for this obj ON THIS FRAME)
            obj_data_on_frame['box'] = None # Clear previous box for this frame
            drawing_data["box_start_pt_img"] = (image_x, image_y)
            drawing_data["temp_box_end_pt_win"] = (x, y)
            drawing_data["drawing_box"] = True
            print(f"  Started box win({x},{y}) -> img({image_x},{image_y}) for object {obj_id_to_modify} on frame {current_fidx}")

    elif event == cv2.EVENT_MOUSEMOVE:
        # Update Box Preview
        if mode == 'b' and drawing_data["drawing_box"] and drawing_data["current_interaction_obj_id"] is not None:
            drawing_data["temp_box_end_pt_win"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish Box
        if mode == 'b' and drawing_data["drawing_box"] and drawing_data["current_interaction_obj_id"] is not None:
            drawing_data["drawing_box"] = False
            obj_id_to_modify = drawing_data["current_interaction_obj_id"]

            # Check if object exists in current frame's interactions
            if obj_id_to_modify in frame_interactions:
                obj_data_on_frame = frame_interactions[obj_id_to_modify]
                start_x_img, start_y_img = drawing_data["box_start_pt_img"]
                end_x_img, end_y_img = window_to_image_coords(x, y)
                x1_img, x2_img = min(start_x_img, end_x_img), max(start_x_img, end_x_img)
                y1_img, y2_img = min(start_y_img, end_y_img), max(start_y_img, end_y_img)

                if x1_img < x2_img and y1_img < y2_img:
                    obj_data_on_frame['box'] = [x1_img, y1_img, x2_img, y2_img]
                    print(f"  Finished box image [{x1_img},{y1_img},{x2_img},{y2_img}] for object {obj_id_to_modify} on frame {current_fidx}")
                else:
                    print(f"  Box definition cancelled (zero size) for object {obj_id_to_modify} on frame {current_fidx}")
                    obj_data_on_frame['box'] = None # Ensure box is None if cancelled
            else:
                print(f"  Warning: Tried to finish box for object {obj_id_to_modify} on frame {current_fidx}, but object data not found for this frame.")

            # Reset temporary box points regardless
            drawing_data["box_start_pt_img"] = None
            drawing_data["temp_box_end_pt_win"] = None


# --- NEW: GUI Function for Multi-Frame Prompting ---
def select_multi_frame_prompts_gui(temp_gui_frame_dir, total_frames):
    """Opens GUI allowing prompting/refinement across multiple frames."""
    global drawing_data, HEADER_HEIGHT
    print("\n--- Interactive Multi-Frame Prompt Selection ---")
    print(" Instructions:")
    print("  - Slider: Navigate between frames.")
    print("  - Buttons: Click buttons in the header area.")
    print("  - Image Area:")
    print("    - L-Click/Drag: Add point or draw box for the ACTIVE object ID on the CURRENT frame.")
    print("    - Mouse Wheel: Zoom In / Out.")
    print("    - Middle-Click + Drag: Pan the view.")
    print("  - Keyboard Shortcuts:")
    print("    - P: Positive Point | N: Negative Point | B: Box")
    print("    - O: Start defining a NEW object ID (use BEFORE first click for the object).")
    print("    - R: Reset ALL prompts on ALL frames and view.")
    print("    - F / Enter / Q: Finish prompt selection.")
    print("--------------------------------------------------\n")

    # --- Initialize GUI state ---
    drawing_data["current_frame_index"] = 0
    drawing_data["total_frames_for_gui"] = total_frames
    drawing_data["temp_frame_dir_gui"] = temp_gui_frame_dir
    drawing_data["base_frames_gui"] = {} # Clear frame cache
    drawing_data["interactions_by_frame"] = {} # Clear previous interactions
    drawing_data["next_global_id"] = 1
    drawing_data["current_class_index"] = 0
    drawing_data["current_mode"] = 'p'
    drawing_data["is_adding_to_new_object"] = True
    drawing_data["current_interaction_obj_id"] = None
    drawing_data["drawing_box"] = False
    drawing_data["box_start_pt_img"] = None
    drawing_data["temp_box_end_pt_win"] = None
    drawing_data["exit_flag"] = False
    drawing_data["zoom_level"] = 1.0
    drawing_data["view_offset_x"] = 0.0
    drawing_data["view_offset_y"] = 0.0
    drawing_data["panning"] = False

    # --- Load the very first frame initially ---
    first_frame_path = os.path.join(temp_gui_frame_dir, f"{0:05d}.jpg")
    if not os.path.exists(first_frame_path):
        print(f"Error: Cannot find first frame for GUI: {first_frame_path}")
        return None
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Error: Could not read first frame image: {first_frame_path}")
        return None
    drawing_data["base_frames_gui"][0] = frame.copy() # Cache frame 0
    frame_height, frame_width = frame.shape[:2]

    # --- Initialize GUI dimensions and layout ---
    drawing_data["base_image_width"] = frame_width
    drawing_data["base_image_height"] = frame_height
    define_buttons(frame_width) # Recalculates HEADER_HEIGHT based on trackbar etc.
    drawing_data["window_width"] = frame_width
    drawing_data["window_height"] = frame_height + HEADER_HEIGHT
    drawing_data["image_area_height"] = frame_height # Initial estimate

    window_title = drawing_data["window_name"]
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, mouse_callback)
    # Create trackbar AFTER namedWindow
    if total_frames > 1:
        cv2.createTrackbar('Frame', window_title, 0, total_frames - 1, on_trackbar)
    cv2.resizeWindow(window_title, frame_width, frame_height + HEADER_HEIGHT)

    # --- Main GUI Loop ---
    while not drawing_data["exit_flag"]:
        # --- Load Current Frame if not cached ---
        current_fidx = drawing_data["current_frame_index"]
        if current_fidx not in drawing_data["base_frames_gui"]:
            frame_path = os.path.join(drawing_data["temp_frame_dir_gui"], f"{current_fidx:05d}.jpg")
            if os.path.exists(frame_path):
                loaded_frame = cv2.imread(frame_path)
                if loaded_frame is not None:
                    # Simple cache: Keep only last 5 frames? Adjust as needed.
                    if len(drawing_data["base_frames_gui"]) > 5:
                       # Remove oldest cached frame (simplistic) - Requires ordered dict or similar for better cache eviction
                       oldest_key = min(drawing_data["base_frames_gui"].keys())
                       try:
                           del drawing_data["base_frames_gui"][oldest_key]
                       except KeyError: pass # Ignore if key already gone
                    drawing_data["base_frames_gui"][current_fidx] = loaded_frame.copy()
                else:
                    print(f"Warning: Could not read frame {current_fidx} from {frame_path}")
                    # Optionally display a placeholder image or the last good frame
                    # For now, try to use frame 0 if current fails
                    if 0 in drawing_data["base_frames_gui"]:
                         current_fidx = 0 # Revert to frame 0 if load fails
                    else: # Should not happen if frame 0 loaded ok initially
                         print("Error: Cannot display any frame.")
                         break # Exit loop if critical failure
            else:
                print(f"Warning: Frame file not found: {frame_path}")
                if 0 in drawing_data["base_frames_gui"]:
                     current_fidx = 0 # Revert to frame 0 if file missing
                else:
                     print("Error: Cannot display any frame.")
                     break # Exit loop if critical failure


        # --- Get the frame to display ---
        base_frame_full_res = drawing_data["base_frames_gui"].get(current_fidx)
        if base_frame_full_res is None:
             print(f"Error: Frame {current_fidx} not loaded or in cache.")
             # Attempt to recover using frame 0 if possible
             base_frame_full_res = drawing_data["base_frames_gui"].get(0)
             if base_frame_full_res is None:
                 print("Fatal Error: Cannot get any base frame for display.")
                 break # Cannot continue

        # --- Window/Viewport Calculations (mostly unchanged) ---
        # Get current window size (might change if user resized)
        # Note: getWindowImageRect might still be unreliable. Assume initial size for offsets.
        win_w = drawing_data["window_width"]
        win_h = drawing_data["window_height"]
        img_area_h = win_h - HEADER_HEIGHT
        drawing_data["image_area_height"] = img_area_h # Update

        view_w_img = win_w / drawing_data["zoom_level"]
        view_h_img = img_area_h / drawing_data["zoom_level"]

        # Clamp offsets
        max_offset_x = max(0.0, drawing_data["base_image_width"] - view_w_img)
        max_offset_y = max(0.0, drawing_data["base_image_height"] - view_h_img)
        drawing_data["view_offset_x"] = max(0.0, min(drawing_data["view_offset_x"], max_offset_x))
        drawing_data["view_offset_y"] = max(0.0, min(drawing_data["view_offset_y"], max_offset_y))

        src_x = int(round(drawing_data["view_offset_x"]))
        src_y = int(round(drawing_data["view_offset_y"]))
        src_w = int(round(view_w_img))
        src_h = int(round(view_h_img))
        src_w = min(src_w, drawing_data["base_image_width"] - src_x)
        src_h = min(src_h, drawing_data["base_image_height"] - src_y)

        # --- Prepare display frame ---
        frame_display = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        frame_display.fill(60)

        if src_w > 0 and src_h > 0:
            viewport = base_frame_full_res[src_y:src_y + src_h, src_x:src_x + src_w]
            try:
                resized_viewport = cv2.resize(viewport, (win_w, img_area_h), interpolation=cv2.INTER_LINEAR)
                frame_display[HEADER_HEIGHT:, :, :] = resized_viewport
            except Exception as e:
                 print(f"Error resizing viewport: {e}")
                 frame_display[HEADER_HEIGHT:, :, :].fill(20) # Indicate error
        else:
            # print("Warning: Viewport size is zero.")
            frame_display[HEADER_HEIGHT:, :, :].fill(30) # Indicate issue

        # --- Draw GUI elements ---
        draw_buttons_and_info(frame_display) # Draw header info, trackbar placeholder, buttons
        draw_prompts_for_current_frame(frame_display[HEADER_HEIGHT:, :, :]) # Draw prompts for current frame

        # --- Show frame and handle keys ---
        cv2.imshow(window_title, frame_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 13 or key == ord('f'): handle_finish()
        elif key == ord('p'): handle_set_mode('p')
        elif key == ord('n'): handle_set_mode('n')
        elif key == ord('b'): handle_set_mode('b')
        elif key == ord('o'): handle_new_object()
        elif key == ord('r'): handle_reset()
        # Add key bindings for frame navigation? (e.g., left/right arrow)
        elif key == 81 or key == ord(','): # Left arrow or comma
            new_fidx = max(0, drawing_data["current_frame_index"] - 1)
            cv2.setTrackbarPos('Frame', window_title, new_fidx)
            on_trackbar(new_fidx) # Manually trigger update
        elif key == 83 or key == ord('.'): # Right arrow or period
            new_fidx = min(drawing_data["total_frames_for_gui"] - 1, drawing_data["current_frame_index"] + 1)
            cv2.setTrackbarPos('Frame', window_title, new_fidx)
            on_trackbar(new_fidx) # Manually trigger update

    cv2.destroyAllWindows()
    print("\n--- Multi-Frame Prompt Selection Finished ---")

    # --- Final Validation/Cleanup of Prompts ---
    final_interactions = {}
    valid_prompts_found = False
    total_prompted_objects = set() # Track unique object IDs prompted

    for frame_idx, frame_data in drawing_data["interactions_by_frame"].items():
        final_frame_interactions = {}
        for obj_id, data in frame_data.items():
            has_points = data.get('points') and len(data['points']) > 0
            has_box = data.get('box') is not None
            if has_points or has_box:
                final_frame_interactions[obj_id] = data
                valid_prompts_found = True
                total_prompted_objects.add(obj_id)
                # Optional: print details per frame
                # print(f"  Frame {frame_idx}, ID: {obj_id}, Class: {data['class']}, Points: {len(data.get('points', []))}, Box: {'Yes' if has_box else 'No'}")
            # else: # Don't print skipping for frames where an object ID exists but has no prompts defined *on that specific frame*
            #     pass
        if final_frame_interactions: # Only add frame to final dict if it has valid prompts
             final_interactions[frame_idx] = final_frame_interactions


    if not valid_prompts_found:
        print("Warning: No valid prompts were collected for any object on any frame.")
        return None # Return None if no objects have valid prompts

    print(f"Collected valid prompts for {len(total_prompted_objects)} unique objects across {len(final_interactions)} frames.")
    # Return the entire structure {frame_idx: {obj_id: data}}
    return final_interactions


# --- NEW: Frame Extraction Function (For GUI) ---
def extract_all_frames_for_gui(video_path, output_dir, target_w=None, target_h=None):
    """
    Extracts ALL frames, resizes them, saves to output_dir FOR GUI USE.
    Returns success, fps, orig_w, orig_h, proc_w, proc_h, count.
    WARNING: Can use significant disk space.
    """
    print(f"Extracting & Resizing ALL frames from {video_path} to {output_dir} for GUI...")
    print("WARNING: This may require significant disk space.")
    if os.path.exists(output_dir):
        print("Temporary GUI frame directory exists. Removing it.")
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Error removing existing directory {output_dir}: {e}")
            return False, 0, 0, 0, 0, 0, 0
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return False, 0, 0, 0, 0, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False, 0, 0, 0, 0, 0, 0

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    proc_width, proc_height = orig_width, orig_height

    if orig_width <= 0 or orig_height <= 0:
        print("Warning: Video properties reported zero dimensions. Cannot proceed.")
        cap.release()
        return False, fps, 0, 0, 0, 0, 0

    aspect_ratio = orig_width / orig_height if orig_height > 0 else 1.0

    if target_w is not None or target_h is not None:
        if target_w is not None and target_h is None:
            proc_width = target_w
            proc_height = int(target_w / aspect_ratio) if aspect_ratio > 0 else 0
        elif target_w is None and target_h is not None:
            proc_height = target_h
            proc_width = int(target_h * aspect_ratio)
        elif target_w is not None and target_h is not None:
            proc_width = target_w
            proc_height = target_h

        proc_width = max(1, int(proc_width))
        proc_height = max(1, int(proc_height))
        if proc_height == 0: # Handle potential zero height after calculation
             print("Error: Calculated processing height is zero.")
             cap.release()
             return False, fps, orig_width, orig_height, proc_width, 0, 0

    resize_enabled = (proc_width != orig_width or proc_height != orig_height)
    if resize_enabled:
        print(f"Resizing frames from {orig_width}x{orig_height} to {proc_width}x{proc_height} for GUI.")
    else:
        print("No resizing needed for GUI frames. Using original resolution.")

    extraction_start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"Warning: Skipped reading invalid frame {frame_count}")
            continue

        if resize_enabled:
            try:
                interpolation = cv2.INTER_AREA if (proc_width < orig_width or proc_height < orig_height) else cv2.INTER_LINEAR
                processed_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=interpolation)
            except Exception as resize_e:
                print(f"Error resizing frame {frame_count}: {resize_e}. Skipping frame.")
                continue
        else:
            processed_frame = frame

        if processed_frame is not None:
            frame_filename = os.path.join(output_dir, f"{frame_count:05d}.jpg")
            write_success = cv2.imwrite(frame_filename, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not write_success:
                print(f"Warning: Failed to write frame {frame_count} to {frame_filename}")
                continue
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Extracted {frame_count}/{total_vid_frames} frames...")

    cap.release()

    if frame_count == 0:
        print("Error: No frames were successfully extracted/resized for GUI.")
        if os.path.exists(output_dir): shutil.rmtree(output_dir)
        return False, fps, orig_width, orig_height, proc_width, proc_height, 0

    if fps is None or fps <= 0:
        print(f"Warning: Invalid FPS ({fps}) detected, defaulting to 25.0")
        fps = 25.0

    print(f"GUI Frame Extraction complete: {frame_count} frames saved in {time.time() - extraction_start_time:.2f}s.")
    return True, fps, orig_width, orig_height, proc_width, proc_height, frame_count


# --- Frame Extraction Function (For Chunking - Unchanged) ---
def extract_frames_range(video_path, output_dir, start_frame, end_frame, target_w=None, target_h=None):
    """Extracts and resizes frames from a specific range in the video (used by chunk processing)."""
    # print(f"  Extracting frames {start_frame}-{end_frame-1} for chunk...") # Less verbose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video {video_path} for range extraction.")
        return False, 0, 0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if orig_width <= 0 or orig_height <= 0:
        print(f"  Error: Invalid original video dimensions ({orig_width}x{orig_height}).")
        cap.release()
        return False, 0, 0

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_vid_frames)
    if start_frame >= end_frame:
        # print(f"  Info: Invalid/Empty frame range ({start_frame}-{end_frame}).") # Can happen if chunk starts at end
        cap.release()
        return False, orig_width, orig_height # Indicate failure but return original dims

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(current_pos - start_frame) > 5: # Allow some leeway
        print(f"  Warning: Could not accurately seek to frame {start_frame} (got {current_pos}). Extraction might be offset.")

    proc_width, proc_height = orig_width, orig_height
    resize_enabled = False
    if target_w is not None or target_h is not None:
        if orig_height <= 0:
            print(f"  Error: Cannot calculate aspect ratio, original height is zero.")
            cap.release()
            return False, orig_width, orig_height
        aspect_ratio = orig_width / orig_height
        if target_w is not None and target_h is None:
            proc_width = target_w
            proc_height = int(target_w / aspect_ratio)
        elif target_w is None and target_h is not None:
            proc_height = target_h
            proc_width = int(target_h * aspect_ratio)
        elif target_w is not None and target_h is not None:
            proc_width = target_w
            proc_height = target_h
        proc_width = max(1, int(proc_width))
        proc_height = max(1, int(proc_height))
        if proc_height <= 0:
             print(f"  Error: Calculated proc_height is zero.")
             cap.release()
             return False, orig_width, orig_height

        if proc_width != orig_width or proc_height != orig_height:
            resize_enabled = True

    frame_count_in_range = 0
    for i in range(end_frame - start_frame):
        actual_frame_index = start_frame + i
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Failed to read frame at index {actual_frame_index} (relative index {i}). End of stream or error.")
            break

        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"  Warning: Skipped invalid frame read at index {actual_frame_index}")
            continue

        if resize_enabled:
            try:
                interpolation = cv2.INTER_AREA if (proc_width < orig_width or proc_height < orig_height) else cv2.INTER_LINEAR
                processed_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=interpolation)
            except Exception as e:
                print(f"  Error resizing frame {actual_frame_index}: {e}. Skipping.")
                continue
        else:
            processed_frame = frame

        frame_filename = os.path.join(output_dir, f"{i:05d}.jpg")
        write_success = cv2.imwrite(frame_filename, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not write_success:
            print(f"  Warning: Failed to write frame {i} (original index {actual_frame_index}) to {frame_filename}")
            continue
        frame_count_in_range += 1

    cap.release()
    # print(f"  Finished extraction for range. Saved {frame_count_in_range} frames.") # Less verbose
    return frame_count_in_range > 0, proc_width, proc_height


# --- Data Saving Helper Functions ---
def get_bbox_from_mask(mask_bool):
    """Calculates the bounding box [x_min, y_min, x_max, y_max] from a boolean mask."""
    # (Unchanged - keeping original logic)
    if mask_bool is None or not isinstance(mask_bool, np.ndarray) or mask_bool.ndim != 2 or mask_bool.size == 0:
        return None
    if not mask_bool.any():
        return None # No True values in mask
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add +1 to max because slicing is exclusive, bbox is inclusive (convention)
    # Ensure max >= min for single pixel masks, but output coords should reflect the single pixel
    # Let's stick to convention: xmax/ymax are the coords of the bottom-right pixel
    return [int(cmin), int(rmin), int(cmax), int(rmax)] # xmin, ymin, xmax, ymax

def convert_prompts_for_json(prompts_by_frame_dict):
    """Converts numpy arrays/lists in the multi-frame prompts dict for JSON."""
    serializable_prompts = {}
    if not prompts_by_frame_dict:
        return serializable_prompts

    for frame_idx_str, frame_data in prompts_by_frame_dict.items():
        # Ensure frame_idx is treated as string key for JSON if it wasn't already
        frame_idx_key = str(frame_idx_str)
        serializable_frame_data = {}
        if not isinstance(frame_data, dict):
             print(f"Warning: Skipping non-dict data for frame_idx {frame_idx_key} in JSON conversion.")
             continue

        for obj_id_str, data in frame_data.items():
             obj_id_key = str(obj_id_str) # Ensure obj_id key is string
             serializable_data = {}
             if not isinstance(data, dict):
                 print(f"Warning: Skipping non-dict data for obj_id {obj_id_key} on frame {frame_idx_key} in JSON conversion.")
                 continue

             serializable_data['class'] = data.get('class', 'Unknown')

             # Convert points
             if 'points' in data and data['points'] is not None:
                 try:
                     serializable_data['points'] = np.array(data['points']).tolist()
                 except Exception as e:
                     print(f"Warning: Could not serialize points for obj {obj_id_key}, frame {frame_idx_key}: {e}")
                     serializable_data['points'] = []
             else:
                 serializable_data['points'] = []

             # Convert labels
             if 'labels' in data and data['labels'] is not None:
                 try:
                     serializable_data['labels'] = np.array(data['labels']).tolist()
                 except Exception as e:
                     print(f"Warning: Could not serialize labels for obj {obj_id_key}, frame {frame_idx_key}: {e}")
                     serializable_data['labels'] = []
             else:
                 serializable_data['labels'] = []

             # Convert box
             if 'box' in data and data['box'] is not None:
                 try:
                     serializable_data['box'] = np.array(data['box']).tolist()
                 except Exception as e:
                     print(f"Warning: Could not serialize box for obj {obj_id_key}, frame {frame_idx_key}: {e}")
                     serializable_data['box'] = None
             else:
                 serializable_data['box'] = None

             serializable_frame_data[obj_id_key] = serializable_data
        serializable_prompts[frame_idx_key] = serializable_frame_data

    return serializable_prompts


# --- MODIFIED: Process Video in Chunks with Multi-Frame Prompt Consideration ---
def process_video_in_chunks(predictor, input_video, temp_dir, output_dir, collected_prompts_by_frame, chunk_size=50):
    """
    Process video in chunks, using collected GUI prompts at chunk starts,
    otherwise using BBox pass-through as fallback.
    """
    if not ENABLE_CHUNKING:
        print("Error: process_video_in_chunks called but ENABLE_CHUNKING is False.")
        return False

    print(f"Processing video in chunks of {chunk_size} frames...")
    print("-> Using GUI prompts if defined for chunk start frame.")
    print("-> Using BBox from previous chunk end as fallback.")
    print("-> NOTE: GUI refinement points defined *mid-chunk* are NOT used for correction during propagation.")

    # --- Get video properties ---
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened(): print(f"Error: Could not open video {input_video}"); return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if total_frames <= 0: print(f"Error: Video contains zero frames."); return False
    if orig_width <= 0 or orig_height <= 0: print(f"Error: Invalid video dimensions ({orig_width}x{orig_height})"); return False
    if fps <= 0: print(f"Warning: Invalid video FPS ({fps}), defaulting to 25.0"); fps = 25.0

    # --- Determine processing dimensions ---
    proc_width, proc_height = orig_width, orig_height
    resize_needed_for_processing = False
    if TARGET_WIDTH is not None or TARGET_HEIGHT is not None:
        if orig_height <= 0: print("Error: Cannot resize, original height is zero."); return False
        aspect_ratio = orig_width / orig_height
        if TARGET_WIDTH is not None and TARGET_HEIGHT is None:
            proc_width = TARGET_WIDTH; proc_height = int(TARGET_WIDTH / aspect_ratio)
        elif TARGET_WIDTH is None and TARGET_HEIGHT is not None:
            proc_height = TARGET_HEIGHT; proc_width = int(TARGET_HEIGHT * aspect_ratio)
        elif TARGET_WIDTH is not None and TARGET_HEIGHT is not None:
            proc_width = TARGET_WIDTH; proc_height = TARGET_HEIGHT
        proc_width = max(1, int(proc_width)); proc_height = max(1, int(proc_height))
        if proc_height <= 0: print("Error: Calculated processing height is zero."); return False
        if proc_width != orig_width or proc_height != orig_height:
            resize_needed_for_processing = True
            print(f"Processing will use resolution: {proc_width}x{proc_height}")
        else: print(f"Processing at original resolution: {proc_width}x{proc_height}")
    else: print(f"Processing at original resolution: {proc_width}x{proc_height}")

    # --- Setup ---
    chunk_frame_dir = os.path.join(temp_dir, 'chunk_processing_frames') # Dedicated dir for chunk processing
    if os.path.exists(chunk_frame_dir): shutil.rmtree(chunk_frame_dir)
    os.makedirs(chunk_frame_dir)

    hdf5_path = os.path.join(output_dir, OUTPUT_HDF5_FILE)
    video_path = os.path.join(output_dir, OUTPUT_VIDEO_NAME)

    # Store BBoxes from the *last* frame of the previous chunk as fallback prompts
    fallback_prompts_from_prev_chunk = {} # {obj_id: {'box': [x1,y1,x2,y2]}}
    # Get a set of all object IDs ever prompted in the GUI
    all_prompted_obj_ids = set()
    object_id_to_class = {} # Map unique obj_id to its class name
    if collected_prompts_by_frame:
        for frame_idx, frame_data in collected_prompts_by_frame.items():
            for obj_id, data in frame_data.items():
                 all_prompted_obj_ids.add(obj_id)
                 if obj_id not in object_id_to_class and 'class' in data:
                      object_id_to_class[obj_id] = data['class']

    print(f"Found {len(all_prompted_obj_ids)} unique object IDs prompted in the GUI.")
    # Ensure all objects have a class mapping, default if needed
    for obj_id in all_prompted_obj_ids:
        if obj_id not in object_id_to_class:
            object_id_to_class[obj_id] = 'Unknown'


    overall_success = True
    start_chunking_time = time.time()

    # --- Setup HDF5 file and Video Writer ---
    try:
        with h5py.File(hdf5_path, 'w') as hf:
            hf.attrs['source_video'] = input_video
            hf.attrs['num_frames'] = total_frames; hf.attrs['fps'] = fps
            hf.attrs['original_height'] = orig_height; hf.attrs['original_width'] = orig_width
            hf.attrs['processing_height'] = proc_height; hf.attrs['processing_width'] = proc_width
            hf.attrs['total_objects_prompted'] = len(all_prompted_obj_ids)
            hf.attrs['chunk_size'] = chunk_size; hf.attrs['chunking_strategy'] = 'gui_prompt_override_bbox_fallback'
            frames_group = hf.create_group('frames')

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (orig_width, orig_height))
            if not video_writer.isOpened(): raise IOError(f"Could not open video writer at {video_path}")

            # --- Process in chunks ---
            current_chunk_start_frame = 0
            while current_chunk_start_frame < total_frames:
                chunk_start = current_chunk_start_frame
                chunk_end = min(chunk_start + chunk_size, total_frames)
                num_frames_in_chunk = chunk_end - chunk_start
                print(f"\n--- Processing Chunk: Frames {chunk_start} to {chunk_end-1} (Size: {num_frames_in_chunk}) ---")

                if num_frames_in_chunk <= 0:
                    print("  Skipping empty chunk.")
                    break # Should not happen with < total_frames check, but safety

                # --- 1. Extract frames for this chunk ---
                # print(f"  Clearing previous chunk frames...") # Less verbose
                if os.path.exists(chunk_frame_dir): # Clear contents, not dir itself
                    for f in os.listdir(chunk_frame_dir): os.remove(os.path.join(chunk_frame_dir, f))
                else: os.makedirs(chunk_frame_dir) # Should exist, but safety

                extract_success, chunk_proc_w, chunk_proc_h = extract_frames_range(
                    input_video, chunk_frame_dir, chunk_start, chunk_end,
                    target_w=proc_width, target_h=proc_height)

                if not extract_success or chunk_proc_w != proc_width or chunk_proc_h != proc_height:
                    print(f"  ERROR: Failed to extract frames correctly for chunk {chunk_start}-{chunk_end-1}. Stopping.")
                    overall_success = False; break
                extracted_count = len(os.listdir(chunk_frame_dir))
                if extracted_count == 0:
                     print(f"  WARNING: Zero frames extracted for non-empty chunk range {chunk_start}-{chunk_end-1}. Skipping chunk.")
                     current_chunk_start_frame = chunk_end # Move to next potential chunk
                     continue
                elif extracted_count != num_frames_in_chunk:
                    print(f"  WARNING: Expected {num_frames_in_chunk} frames, extracted {extracted_count}. Processing extracted count.")
                    num_frames_in_chunk = extracted_count # Adjust for actual extracted frames

                # --- 2. Clear GPU memory ---
                # print("  Clearing GPU cache...") # Less verbose
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                # --- 3. Initialize predictor state ---
                inference_state = None
                try:
                    # print(f"  Initializing inference state...") # Less verbose
                    inference_state = predictor.init_state(video_path=chunk_frame_dir)
                    if inference_state is None: raise ValueError("init_state returned None")

                    # --- 4. Determine & Add Prompts for First Frame OF THIS CHUNK ---
                    print(f"  Determining prompts for frame {chunk_start}...")
                    # Get user prompts specifically defined for the starting frame of this chunk
                    gui_prompts_for_chunk_start = collected_prompts_by_frame.get(chunk_start, {})
                    prompts_to_add_this_chunk = {} # {obj_id: {'points':np, 'labels':np, 'box':np}}

                    # Determine which objects *could* be tracked in this chunk
                    # Start with objects potentially passed from the previous chunk
                    potential_obj_ids = set(fallback_prompts_from_prev_chunk.keys())
                    # Add any objects that have their *first* prompt on this frame
                    potential_obj_ids.update(gui_prompts_for_chunk_start.keys())
                    # Ensure we only consider objects originally prompted
                    potential_obj_ids = potential_obj_ids.intersection(all_prompted_obj_ids)


                    for obj_id in potential_obj_ids:
                        points_np, labels_np, box_np = None, None, None
                        prompt_source = "None"

                        # --- Check for GUI prompts defined SPECIFICALLY for chunk_start ---
                        if obj_id in gui_prompts_for_chunk_start:
                            gui_prompt_data = gui_prompts_for_chunk_start[obj_id]
                            # Prioritize box if defined by user on this frame
                            if gui_prompt_data.get('box') is not None:
                                try:
                                    box_np = np.array(gui_prompt_data['box'], dtype=np.float32)
                                    if box_np.shape == (4,): prompt_source = f"GUI Box (Frame {chunk_start})"
                                    else: box_np = None # Invalid shape
                                except Exception: box_np = None
                            # If no GUI box, check for GUI points on this frame
                            if box_np is None and gui_prompt_data.get('points'):
                                try:
                                    points_list = gui_prompt_data['points']
                                    labels_list = gui_prompt_data['labels']
                                    if points_list and labels_list and len(points_list) == len(labels_list):
                                        points_np = np.array(points_list, dtype=np.float32)
                                        labels_np = np.array(labels_list, dtype=np.int32)
                                        # Basic validation
                                        if not (points_np.ndim == 2 and points_np.shape[1] == 2 and \
                                                labels_np.ndim == 1 and labels_np.shape[0] == points_np.shape[0]):
                                            points_np, labels_np = None, None
                                        else:
                                             prompt_source = f"GUI Points (Frame {chunk_start})"
                                    else: points_np, labels_np = None, None
                                except Exception: points_np, labels_np = None, None

                        # --- If NO GUI prompt for this object on this frame, use fallback BBox ---
                        if prompt_source == "None" and obj_id in fallback_prompts_from_prev_chunk:
                            fallback_data = fallback_prompts_from_prev_chunk[obj_id]
                            if fallback_data.get('box') is not None:
                                try:
                                    box_np = np.array(fallback_data['box'], dtype=np.float32)
                                    if box_np.shape == (4,): prompt_source = f"Fallback BBox (Prev Chunk)"
                                    else: box_np = None
                                except Exception: box_np = None

                        # --- If we have a valid prompt (GUI or Fallback), prepare to add it ---
                        if prompt_source != "None":
                            prompts_to_add_this_chunk[obj_id] = {'points': points_np, 'labels': labels_np, 'box': box_np, 'source': prompt_source}


                    # --- Add the collected prompts to the predictor ---
                    prompts_added_count = 0
                    if not prompts_to_add_this_chunk:
                         print("  WARNING: No objects with valid prompts (GUI or Fallback) found for this chunk start. Skipping chunk.")
                    else:
                        print(f"  Adding {len(prompts_to_add_this_chunk)} prompts:")
                        for obj_id, add_data in prompts_to_add_this_chunk.items():
                            try:
                                # print(f"    Obj {obj_id}: Source = {add_data['source']}") # Verbose
                                predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=0, # Always frame 0 *within the chunk's frame files*
                                    obj_id=obj_id,
                                    points=add_data['points'],
                                    labels=add_data['labels'],
                                    box=add_data['box']
                                )
                                prompts_added_count += 1
                            except Exception as e:
                                print(f"    ERROR adding prompt for obj {obj_id}: {e}")

                        if prompts_added_count == 0:
                             print("  WARNING: Failed to add any prompts to predictor state. Skipping chunk.")
                        else:
                             print(f"  Successfully added {prompts_added_count} prompts.")


                    # --- 5. Process frames if prompts were added ---
                    if prompts_added_count > 0:
                        print(f"  Propagating masks through {num_frames_in_chunk} frames...")
                        chunk_video_segments = {} # {rel_frame_idx: {obj_id: mask_bool_proc}}
                        chunk_video_centroids = {}# {rel_frame_idx: {obj_id: centroid_proc}}
                        chunk_processing_start_time = time.time()
                        objects_lost_in_chunk = set() # Track objects that disappear

                        with torch.no_grad(), autocast(enabled=(torch.cuda.is_available())):
                            prop_iter = predictor.propagate_in_video(inference_state)
                            processed_rel_frame_indices = set() # Track which frames were actually processed

                            for rel_frame_idx, out_obj_ids, out_mask_logits in prop_iter:
                                processed_rel_frame_indices.add(rel_frame_idx)
                                abs_frame_idx = chunk_start + rel_frame_idx
                                frame_masks_proc = {}; frame_centroids_proc = {}

                                currently_tracked_ids = set(out_obj_ids).intersection(all_prompted_obj_ids)

                                if len(currently_tracked_ids) > 0 and out_mask_logits is not None:
                                    try:
                                        # Get masks corresponding to the tracked IDs
                                        mask_indices = [i for i, obj_id in enumerate(out_obj_ids) if obj_id in currently_tracked_ids]
                                        if not mask_indices: continue # No relevant objects in output

                                        if out_mask_logits.ndim > 3 and out_mask_logits.shape[1] == 1:
                                            masks_t = (out_mask_logits[mask_indices].squeeze(1) > 0.0)
                                        elif out_mask_logits.ndim == 3:
                                            masks_t = (out_mask_logits[mask_indices] > 0.0)
                                        else: raise ValueError(f"Unexpected mask logit shape: {out_mask_logits.shape}")

                                        masks_np_proc = masks_t.cpu().numpy() # [N, H, W] at proc_res

                                        # --- Calculate centroids ---
                                        N, H, W = masks_t.shape
                                        if N > 0 and H == proc_height and W == proc_width:
                                            masks_float = masks_t.float()
                                            area = masks_float.sum(dim=(1,2)) + 1e-6
                                            device = masks_t.device
                                            xs_vec = torch.arange(W, device=device, dtype=torch.float).view(1, 1, W)
                                            ys_vec = torch.arange(H, device=device, dtype=torch.float).view(1, H, 1)
                                            cx = (masks_float * xs_vec).sum(dim=(1,2)) / area
                                            cy = (masks_float * ys_vec).sum(dim=(1,2)) / area
                                            centroids_np_proc = torch.stack([cx, cy], dim=1).cpu().numpy()
                                        else: centroids_np_proc = np.empty((0, 2), dtype=np.float32)

                                        # Store results per object for this frame
                                        obj_idx_map = {obj_id: i for i, obj_id in enumerate(currently_tracked_ids)} # Map obj_id to index in masks_t
                                        for i, obj_id in enumerate(currently_tracked_ids):
                                            if i < masks_np_proc.shape[0] and masks_np_proc[i].shape == (proc_height, proc_width):
                                                frame_masks_proc[obj_id] = masks_np_proc[i]
                                            if i < centroids_np_proc.shape[0]:
                                                frame_centroids_proc[obj_id] = centroids_np_proc[i]

                                    except Exception as e:
                                        print(f"    Error processing masks/centroids for frame {abs_frame_idx} (relative {rel_frame_idx}): {e}")

                                chunk_video_segments[rel_frame_idx] = frame_masks_proc
                                chunk_video_centroids[rel_frame_idx] = frame_centroids_proc

                                # Check for objects lost compared to start of chunk
                                # lost_now = set(prompts_to_add_this_chunk.keys()) - currently_tracked_ids
                                # objects_lost_in_chunk.update(lost_now)

                        print(f"  Chunk propagation finished in {time.time() - chunk_processing_start_time:.2f} seconds.")
                        # if objects_lost_in_chunk:
                        #      print(f"  Objects lost during this chunk: {list(objects_lost_in_chunk)}")

                        # --- 6. Save Data & Generate Video Frames for this Chunk ---
                        print(f"  Saving HDF5 data and writing video frames...")
                        scale_w = orig_width / proc_width; scale_h = orig_height / proc_height
                        cap_orig = cv2.VideoCapture(input_video)
                        if not cap_orig.isOpened():
                            print(f"  ERROR: Could not reopen original video. Skipping video output for this chunk.")
                        else:
                            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)

                        # Iterate over the expected range, but use data only if processed
                        for rel_frame_idx in range(num_frames_in_chunk):
                            abs_frame_idx = chunk_start + rel_frame_idx
                            # --- HDF5 Saving ---
                            try:
                                frame_group = frames_group.create_group(f'frame_{abs_frame_idx:05d}')
                                objects_found_count = 0
                                # Only save data if the frame was actually processed and has segments
                                if rel_frame_idx in processed_rel_frame_indices and rel_frame_idx in chunk_video_segments:
                                    for obj_id, mask_bool_proc in chunk_video_segments[rel_frame_idx].items():
                                        if obj_id in all_prompted_obj_ids: # Ensure it's an object we care about
                                            class_name = object_id_to_class.get(obj_id, 'Unknown')
                                            try:
                                                mask_bool_orig = cv2.resize(mask_bool_proc.astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                                                if mask_bool_orig.shape != (orig_height, orig_width): continue # Skip save if resize failed

                                                obj_dset = frame_group.create_dataset(f'object_{obj_id:03d}_mask', data=mask_bool_orig, dtype='bool', compression='gzip', chunks=(min(orig_height, 256), min(orig_width, 256)))
                                                obj_dset.attrs['global_id'] = obj_id
                                                obj_dset.attrs['class_name'] = class_name

                                                # Centroid
                                                if rel_frame_idx in chunk_video_centroids and obj_id in chunk_video_centroids[rel_frame_idx]:
                                                    cx_proc, cy_proc = chunk_video_centroids[rel_frame_idx][obj_id]
                                                    obj_dset.attrs['centroid_x'] = cx_proc * scale_w
                                                    obj_dset.attrs['centroid_y'] = cy_proc * scale_h
                                                # BBox
                                                bbox_proc = get_bbox_from_mask(mask_bool_proc)
                                                if bbox_proc:
                                                    x1p, y1p, x2p, y2p = bbox_proc
                                                    bbox_orig = [int(x1p*scale_w), int(y1p*scale_h), int(x2p*scale_w), int(y2p*scale_h)]
                                                    obj_dset.attrs['bbox_xyxy'] = bbox_orig
                                                objects_found_count +=1
                                            except Exception as e: print(f"    Error saving mask/attrs for obj {obj_id}, frame {abs_frame_idx}: {e}")
                                frame_group.attrs['objects_found_count'] = objects_found_count
                            except Exception as e: print(f"  Error creating HDF5 group/saving for frame {abs_frame_idx}: {e}")

                            # --- Video Frame Writing ---
                            if cap_orig and cap_orig.isOpened():
                                ret_orig, frame_bgr_orig = cap_orig.read()
                                if not ret_orig or frame_bgr_orig is None:
                                    print(f"  Warning: Could not read original frame {abs_frame_idx}. Writing black frame.")
                                    video_writer.write(np.zeros((orig_height, orig_width, 3), dtype=np.uint8))
                                    continue # Skip drawing if frame read failed
                                try:
                                    final_frame = frame_bgr_orig.copy()
                                    # Draw only if frame was processed
                                    if rel_frame_idx in processed_rel_frame_indices and rel_frame_idx in chunk_video_segments:
                                        mask_color_layer = np.zeros_like(final_frame, dtype=np.uint8)
                                        object_ids_in_frame = sorted(chunk_video_segments[rel_frame_idx].keys())
                                        for obj_id in object_ids_in_frame:
                                            if obj_id in chunk_video_segments[rel_frame_idx]:
                                                mask_bool_proc = chunk_video_segments[rel_frame_idx][obj_id]
                                                try:
                                                    mask_bool_orig_draw = cv2.resize(mask_bool_proc.astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_NEAREST).astype(bool)
                                                    if mask_bool_orig_draw.shape == (orig_height, orig_width):
                                                        mask_color_layer[mask_bool_orig_draw] = get_color(obj_id)
                                                except Exception as resize_e: print(f"    Warning: Failed to resize mask for drawing obj {obj_id} frame {abs_frame_idx}: {resize_e}")
                                        alpha = 0.4
                                        cv2.addWeighted(mask_color_layer, alpha, final_frame, 1.0, 0, dst=final_frame)
                                        # Optional Text Labels (Uncomment if needed)
                                        # ... (Add text drawing code here, similar to previous version, using scaled centroids) ...

                                    video_writer.write(final_frame) # Write frame even if not processed (shows original)
                                except Exception as e:
                                    print(f"  Error creating/writing video frame {abs_frame_idx}: {e}")
                                    video_writer.write(np.zeros((orig_height, orig_width, 3), dtype=np.uint8)) # Fallback black frame
                            else: # If cap_orig failed to open earlier
                                 video_writer.write(np.zeros((orig_height, orig_width, 3), dtype=np.uint8))


                        if cap_orig and cap_orig.isOpened(): cap_orig.release()


                        # --- 7. Prepare FALLBACK Prompts (BBoxes) for the NEXT Chunk ---
                        # print(f"  Preparing fallback prompts for next chunk...") # Less verbose
                        prompts_for_following_chunk = {}
                        # Use the last successfully processed frame index within the chunk
                        last_processed_rel_idx = -1
                        if processed_rel_frame_indices:
                             last_processed_rel_idx = max(processed_rel_frame_indices)

                        if last_processed_rel_idx >= 0 and last_processed_rel_idx in chunk_video_segments:
                            last_frame_masks = chunk_video_segments[last_processed_rel_idx]
                            tracked_obj_ids_last_frame = last_frame_masks.keys()
                            # print(f"  Objects tracked in last processed frame ({chunk_start + last_processed_rel_idx}): {list(tracked_obj_ids_last_frame)}") # Debug

                            for obj_id in tracked_obj_ids_last_frame:
                                if obj_id in all_prompted_obj_ids: # Only carry over originally prompted objects
                                    mask_proc = last_frame_masks.get(obj_id)
                                    if mask_proc is not None and mask_proc.any():
                                        bbox_proc = get_bbox_from_mask(mask_proc)
                                        if bbox_proc: # If a valid bbox was found
                                            prompts_for_following_chunk[obj_id] = {'box': bbox_proc} # Store only the box
                                            # print(f"    Generated fallback bbox prompt for obj {obj_id}: {bbox_proc}") # Verbose

                        # Update the fallback prompts for the next iteration
                        fallback_prompts_from_prev_chunk = prompts_for_following_chunk
                        # print(f"  Prepared {len(fallback_prompts_from_prev_chunk)} fallback prompts for next chunk.") # Less verbose

                    # --- End of processing block for chunk ---

                except Exception as e:
                    print(f"  FATAL ERROR processing chunk {chunk_start}-{chunk_end-1}: {e}")
                    import traceback; traceback.print_exc()
                    overall_success = False; break # Stop processing further chunks
                finally:
                    # --- 8. Clean up state for the processed chunk ---
                    if inference_state is not None:
                        # print("  Resetting predictor state.") # Less verbose
                        try: predictor.reset_state(inference_state)
                        except Exception as reset_e: print(f"    Warning: Error resetting predictor state: {reset_e}")
                    # print("  Running garbage collection...") # Less verbose
                    del inference_state
                    if 'chunk_video_segments' in locals(): del chunk_video_segments
                    if 'chunk_video_centroids' in locals(): del chunk_video_centroids
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                # --- Move to the next chunk ---
                current_chunk_start_frame = chunk_end
            # --- End of chunk processing loop ---

            print("\nReleasing video writer...")
            video_writer.release()

    except IOError as e: print(f"IO Error during HDF5/Video setup: {e}"); overall_success = False
    except Exception as e:
        print(f"An unexpected error occurred during chunk processing loop: {e}")
        import traceback; traceback.print_exc(); overall_success = False
        if 'video_writer' in locals() and video_writer.isOpened():
            print("Releasing video writer due to error..."); video_writer.release()

    # --- Final Cleanup ---
    print("Cleaning up temporary chunk frame directory...")
    if os.path.exists(chunk_frame_dir):
        try: shutil.rmtree(chunk_frame_dir)
        except Exception as e: print(f"Warning: Failed to remove chunk frame directory '{chunk_frame_dir}': {e}")

    print(f"Chunk processing finished in {time.time() - start_chunking_time:.2f} seconds.")
    return overall_success


# --- Main Execution ---
def main():
    start_total_time = time.time()
    predictor = None
    temp_gui_frame_dir = os.path.join(TEMP_FRAME_DIR, "gui_frames") # Specific subdir for GUI frames

    # --- 0. Check if Chunking Enabled ---
    if not ENABLE_CHUNKING:
        print("ERROR: This script version requires ENABLE_CHUNKING = True."); sys.exit(1)

    # --- 1. Pre-checks ---
    if not os.path.exists(INPUT_VIDEO_MP4): print(f"Error: Input video not found: '{INPUT_VIDEO_MP4}'"); sys.exit(1)
    if not os.path.exists(SAM2_CHECKPOINT): print(f"Error: SAM2 Checkpoint not found: '{SAM2_CHECKPOINT}'"); sys.exit(1)
    try: from sam2.build_sam import build_sam2_video_predictor
    except ImportError: print("Error: Failed to import 'build_sam2_video_predictor'. Is SAM2 installed?"); sys.exit(1)

    # --- 2. Extract ALL Frames for GUI (Requires Disk Space!) ---
    # Ensure main temp dir exists
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
    extract_success, fps, orig_w, orig_h, proc_w, proc_h, total_frames = extract_all_frames_for_gui(
        INPUT_VIDEO_MP4, temp_gui_frame_dir,
        target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT
    )
    if not extract_success or total_frames <= 0:
        print("Error: Failed to extract frames for GUI.")
        if os.path.exists(TEMP_FRAME_DIR): shutil.rmtree(TEMP_FRAME_DIR)
        sys.exit(1)
    print(f"GUI frames ready at {proc_w}x{proc_h}. Total: {total_frames}.")


    # --- 3. Interactive Multi-Frame Prompt Selection ---
    collected_prompts = select_multi_frame_prompts_gui(temp_gui_frame_dir, total_frames)

    # GUI finished, frame cache no longer needed (but keep dir for now if needed by chunking)
    drawing_data["base_frames_gui"] = {} # Clear cache
    gc.collect()

    if not collected_prompts:
        print("No valid prompts selected or GUI closed prematurely. Exiting.")
        if os.path.exists(TEMP_FRAME_DIR): shutil.rmtree(TEMP_FRAME_DIR)
        sys.exit(0)

    # --- 4. Prepare Output ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_json_path = os.path.join(OUTPUT_DIR, OUTPUT_PROMPTS_JSON)
    print(f"\nOutputs will be saved to: {OUTPUT_DIR}")

    # Save Collected Prompts (coords relative to proc_w, proc_h)
    try:
        prompts_for_json = convert_prompts_for_json(collected_prompts)
        prompts_metadata = {
            "prompt_definition_width": proc_w,
            "prompt_definition_height": proc_h,
            "prompts_by_frame": prompts_for_json # Structure is {frame_idx_str: {obj_id_str: data}}
        }
        with open(output_json_path, 'w') as f: json.dump(prompts_metadata, f, indent=4)
        print(f"Saved collected prompts (coords relative to {proc_w}x{proc_h}) to {output_json_path}")
    except Exception as e: print(f"Warning: Failed to save collected prompts JSON: {e}")

    # --- 5. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if str(device) == 'mps': print("Warning: MPS support is preliminary.")
    print(f"Using device: {device}")

    # --- 6. Load Predictor ---
    print(f"Loading SAM2 video predictor (Config: '{MODEL_CFG}')...")
    load_start_time = time.time()
    try:
        gc.collect(); torch.cuda.empty_cache()
        predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device=device)
    except Exception as e:
        print(f"\nError building predictor: {e}");
        if os.path.exists(TEMP_FRAME_DIR): shutil.rmtree(TEMP_FRAME_DIR)
        sys.exit(1)
    print(f"Predictor loaded in {time.time() - load_start_time:.2f} seconds.")


    # --- 7. Process video using MODIFIED chunking function ---
    processing_success = process_video_in_chunks(
        predictor=predictor,
        input_video=INPUT_VIDEO_MP4,
        temp_dir=TEMP_FRAME_DIR, # Main temp dir, chunk func manages subdirs
        output_dir=OUTPUT_DIR,
        collected_prompts_by_frame=collected_prompts, # Pass the GUI prompts
        chunk_size=CHUNK_SIZE
    )


    # --- 8. Final Cleanup ---
    print("\nCleaning up predictor...")
    del predictor; gc.collect(); torch.cuda.empty_cache()

    # Remove main temporary directory (contains gui_frames and chunk_processing_frames subdirs)
    if os.path.exists(TEMP_FRAME_DIR):
        try:
            shutil.rmtree(TEMP_FRAME_DIR)
            print(f"Removed main temporary directory: {TEMP_FRAME_DIR}")
        except Exception as rm_e: print(f"Warning: Failed to remove main temp dir '{TEMP_FRAME_DIR}': {rm_e}")

    if processing_success: print(f"\nScript finished successfully in {time.time() - start_total_time:.2f} seconds.")
    else: print(f"\nScript finished with ERRORS in {time.time() - start_total_time:.2f} seconds.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


# --- Entry Point ---
if __name__ == "__main__":
    gc.enable()
    main()
