import copy
import enum
import os
import math
import random
import re
import sys
import time

import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog, ttk


########################################################################################################################
# Utility functions ####################################################################################################
########################################################################################################################

# Return a list of filepaths in a directory that match on any of the extensions.
# If recursive, include any matches in all subdirectories.
def get_files(directory, extensions, recursive=False):
    matched_files = []
    if not os.path.isdir(directory) or len(extensions) == 0:
        return matched_files

    def match_files(collection, root):
        for file in collection:
            name, ext = os.path.splitext(file)
            if ext in extensions:
                matched_files.append(os.path.normpath(os.path.join(root, file)))

    if recursive:
        for subdir, _, files in os.walk(directory):
            match_files(files, subdir)
    else:
        match_files(os.listdir(directory), directory)

    return matched_files

# Given a time interval expressed in seconds, convert to a tuple of (hours, minutes, seconds, milliseconds)
def get_hours_minutes_seconds_milliseconds_from_seconds(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = 1000 * (seconds - math.floor(seconds))
    seconds = math.floor(seconds)
    return hours, minutes, seconds, milliseconds

# Given a tuple of (hours, minutes, seconds, milliseconds), return a time string formatted by "HH:MM:SS:MSS".
def get_time_string_from_hh_mm_ss_ms(hours, minutes, seconds, milliseconds):
    # Don't print hours if less than 60 minutes
    if hours > 0:
        time_string = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"
    else:
        time_string = f"{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"
    return time_string

# Given a time interval expressed in seconds, return a time string formatted by "HH:MM:SS:MSS".
def get_time_string_from_seconds(seconds):
    return get_time_string_from_hh_mm_ss_ms(*get_hours_minutes_seconds_milliseconds_from_seconds(seconds))

def valid_resize_factor(factor):
    if factor is None:
        return False
    if len(factor) != 2:
        return False
    for component in factor:
        if not type(component) == float or type(component) == int:
            return False
        if component <= 0:
            return False
    return True

########################################################################################################################

# Video filtering options
class VideoFilter(enum.IntEnum):
    NO_FILTER = 0

    MONOCHROME = 1
    VALUE_INVERT = 2

    RED_FILTER = 3
    GREEN_FILTER = 4
    BLUE_FILTER = 5
    YELLOW_FILTER = 6
    CYAN_FILTER = 7
    MAGENTA_FILTER = 8

    SWAP_RED_GREEN = 9
    SWAP_GREEN_BLUE = 10
    SWAP_BLUE_RED = 11
    CYCLE_BLUE_GREEN_RED_ONCE = 12
    CYCLE_BLUE_GREEN_RED_TWICE = 13

    BGR_TO_HSV = 14
    RGB_TO_HSV = 15
    HSV_TO_BGR = 16
    HSV_TO_RGB = 17
    BGR_TO_HLS = 18
    RGB_TO_HLS = 19
    HLS_TO_BGR = 20
    HLS_TO_RGB = 21
    BGR_TO_LAB = 22
    RGB_TO_LAB = 23
    LAB_TO_BGR = 24
    LAB_TO_RGB = 25
    BGR_TO_LUV = 26
    RGB_TO_LUV = 27
    LUV_TO_BGR = 28
    LUV_TO_RGB = 29

    FILTER_COUNT = 30

class VideoFilterMode(enum.IntEnum):
    NORMAL = 0
    RANDOM = 1

    FILTER_MODES_COUNT = 2

# User options to interrupt normal player functionality
class UserRequest(enum.Enum):
    NONE = 0
    FORCE_PLAY = 1
    ABORT = 2


def display_user_manual():
    user_manual = r'''
    Welcome to the weird and wacky video machine of madness.
    
    After loading video files (see 'L' below), files will 
    start playing randomly (without replacement). Good luck.
    WIP. Many features to be fixed, refined, and added...

    These key-codes (case-sensitive) will trigger the corresponding functions.
    
    L - Select individual files to add to the library matching the given file extensions.
    o - Open a single video file to play immediately and add to the library of files.
    O - Open a folder of video files and add all files matching the given extensions 
        to the video library from this folder and any of its sub-folders.
    q - Exit. Any video recording in progress will be completed.
    n - Skip to the next video.
    Space - Pause/resume video playback. Nice and intuitive.
    Backspace - Jump to the start of the current video. (Use 'delete' on mac.)
    h - Print this help menu, duh.
    i - Print basic video information: resolution, frame rate, current frame number, 
        total number of frames, and the current and total running time in mm:ss:ms format.
    l - Toggle loop mode on/off. When turned on, if a video reaches the end, it will jump back
        to the start and continue looping indefinitely.
    t - Set the current time to recall with 'j'.
    j - Jump back to the time set by 't'.
    g - Toggle random filter mode. When turned on, a random filter will be selected each time a new video is loaded.
    f - Jump forward one second scaled by the current speed factor (see 's' and 'd').
    F - Jump forward five seconds scaled by the current speed factor (see 's' and 'd').
    r - Jump backward one second scaled by the current speed factor (see 's' and 'd').
    R - Jump backward one second scaled by the current speed factor (see 's' and 'd').
    s - Cycle through speed factors (1.0x, 2.0x, 4.0x, 8.0x, 16.0x, 0.0625x, 0.125x, 0.25x, 0.5x) in the 'fast' direction.
    d - Cycle through speed factors (1.0x, 0.5x, 0.25x, 0.125x, 0.0625x, 16.0x, 8.0x, 4.0x, 2.0x) in the 'slow' direction.
    a - Return to normalcy. Set the speed to 1.0x and turn off the filter.
    y - Cycle through video filters in the + direction.
    u - Cycle through video filters in the - direction.
    c - Toggle cutup mode on/off. When turned on, starting on the load of the next video, videos will play starting
        at random locations for the duration specified by 'k'. Combine with random filter mode for some wild results.
    k - Set the cutup duration in milliseconds via an interactive dialog window.
    z - Toggle stream recording to a file. Any frames displayed while the video player is paused will not be written
        to the recording. The file will be saved to the default user video file directory. On Windows, this is
        C:\Users\<user>\Videos; on mac, it's /Users/<user>/Movies. The filename SHOULD be unique since the filename
        contains a string representing the time in seconds since the epoch when the recording file was created.
        TODO: Use the same approach but make the filename in human-readable date format.
    ` - Print out the current list of files in the video library. More of a diagnostic feature since you can't do
        anything with it.
        
          
    -._.=-._.=-._.=-._.=-._.=-._.=-._.=-._
    - The Super Secret Favorites Feature -
    .=-._.=-._.=-._.=-._.=-._.=-._.=-._.=-
    
    This is the extra-special coup de grâce. Excepting the keys already bound to functions as described above,
    any other key press will set the current video as a favorite to be recalled any time by a subsequent press
    of the same key, as long as that key isn't already set to another video (or the same video).
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    Video filters (in order): no filter, monochrome, value invert, red channel, green channel, blue channel,
    yellow channel, cyan channel, magenta channel, RGB to GRB, RGB to RBG, RGB to BGR, RGB to BRG, RGB to GBR,
    BGR to HSV, RGB to HSV, HSV to BGR, HSV to RGB, BGR to HLS, RGB to HLS, HLS to BGR, HLS to RGB,
    BGR to LAB, RGB to LAB, LAB to BGR, LAB to RGB, BGR to LUV, RGB to LUV, LUV to BGR, LUV to RGB
    '''

    user_window = tk.Tk()

    user_window.geometry("1300x1200")
    user_window.resizable(False, False)

    label = tk.Label(user_window, text=user_manual, font=("Courier 12"), justify=tk.LEFT, anchor="center")
    label.pack(padx=10, pady=10)

    button = tk.Button(user_window, text="Got it", font=("Bahnscrift 14"), anchor="center",
                       command=lambda: user_window.destroy())
    button.pack(pady=10)

    user_window.mainloop()


class VideoPlayer:
    def __init__(self, recording_directory=None, recording_dimensions=None):
        self.loop_mode = False
        self.is_paused = False
        self.cutup_mode = False
        self.cutup_interval = 1000

        self.is_recording = False
        if recording_directory is None:
            # Point to the platform-specific user video directory
            if sys.platform == "darwin":
                recording_directory = os.path.expanduser("~/Movies")
            elif sys.platform == "win32":
                recording_directory = os.path.expandvars(r"%USERPROFILE%\Videos")
            else:
                print("Platform unsupported! How are you running this??")
                exit(0)
        self.recording_directory = os.path.normpath(recording_directory)
        self.recording_dimensions = recording_dimensions
        self.recording_filepath = None
        self.recording_capture = None

        self.video_filter_mode = VideoFilterMode.NORMAL
        self.user_request = UserRequest.NONE

        # Keep video file names as keys in a dictionary (with dummy values) to easily avoid adding duplicates
        self.video_files_dict = {}

        # The current video file playing
        self.video_file = None

        self.fps = 0
        self.frame_count = 0
        self.frame_count_digits = 0
        self.frame_time_ms = 0
        self.total_time_ms = 0
        self.current_time_ms = 0
        self.frame_dimensions = (0, 0)

        self.favorites_map = {}

        self.speed_factors = [1.0, 2.0, 4.0, 8.0, 16.0, 0.0625, 0.125, 0.25, 0.5]
        self.speed_factor_index = 0

    # Return the list of video files currently loaded in the video player.
    def video_file_list(self):
        return list(self.video_files_dict.keys())

    # Return the number of video files currently loaded in the video player.
    def video_file_count(self):
        return len(self.video_file_list())

    # Print an indexed list of video files currently loaded in the video player.
    def print_video_file_list(self):
        file_counter = 0
        for video_file in self.video_file_list():
            print(f"{file_counter} {video_file}")
            file_counter += 1

    # Print a string with relevant video properties: width and height in pixels, fps, current frame number,
    # total frame count, current time in hh:mm:ss.mss and total time in hh:mm:ss.ms
    def print_basic_video_properties(self, video_capture):
        if video_capture is None:
            return
        if not video_capture.isOpened():
            return

        # Get the current frame position
        frame_position = int(video_capture.get(cv.CAP_PROP_POS_FRAMES))

        # Compute time format strings
        current_time_string = get_time_string_from_seconds(self.current_time_ms / 1000)
        total_time_string = get_time_string_from_seconds(self.total_time_ms / 1000)

        print(f'{self.video_file}\n{self.frame_dimensions[0]}x{self.frame_dimensions[1]}, {self.fps:0.2f} fps | '
              f'{frame_position:0{self.frame_count_digits}d}/{int(self.frame_count)} | '
              f'{current_time_string}/{total_time_string}')

    def load_videos_from_files(self, files):
        file_count = 0
        for video_file in files:
            self.video_files_dict[video_file] = 0
            file_count += 1
        print(f"Loaded {file_count} video file{'s' if file_count != 1 else ''}.")
        return file_count

    def load_videos_from_directory(self, directory, extensions, recursive=False):
        return self.load_videos_from_files(get_files(directory, extensions, recursive))

    def load_videos_interactive(self, from_directory=False, recursive=True):
        def parse_extensions(extensions):
            extension_pattern = r"\.[^ ]+"
            return re.findall(extension_pattern, extensions)

        def load_videos():
            extensions = parse_extensions(entry.get())
            if len(extensions) == 0:
                print("No valid extensions found.")
                return

            filetypes = map(lambda x: ("Video file", f"*{x}"), extensions)
            if from_directory:
                directory = os.path.normpath(filedialog.askdirectory(mustexist=True, title="Load all videos within a directory."))
                files = get_files(directory, extensions, recursive=recursive)
            else:
                files = filedialog.askopenfilenames(filetypes=filetypes, title="Load video files.")

            if self.load_videos_from_files(files=files) > 0:
                user_window.destroy()
                return

        def handle_return_press(event):
            load_videos()

        # Create an instance of Tkinter frame
        user_window = tk.Tk()

        # Set the geometry of Tkinter frame
        user_window.geometry("500x120")
        user_window.resizable(False, False)

        extensions_label = tk.Label(user_window, text="File extensions, e.g. \".mp4 .mov .avi\"", font=("Bahnscrift", 14))
        extensions_label.pack(pady=10)

        # Create an Entry widget to accept User Input
        entry = tk.Entry(user_window, width=40, font=("Bahnscrift", 14))
        entry.insert(0, ".mp4 .mov .avi")
        entry.bind('<Return>', handle_return_press)
        entry.pack()
        entry.focus_set()

        button = ttk.Button(user_window, text="Load video files", command=load_videos)
        button.pack(pady=10)

        def set_cutup_interval():
            interval = entry.get()
            if not interval.isnumeric():
                user_window.destroy()
                return

            interval = int(interval)
            if interval <= 0:
                user_window.destroy()
                return

            self.cutup_interval = interval
            user_window.destroy()
            print(f"Setting cutup interval to {interval}ms.")

        def handle_return_press(event):
            set_cutup_interval()

        user_window.mainloop()

    def filter_frame(self, frame, video_filter):
        def apply_filter_mask(mask, frame_to_filter):
            filter_matrix = np.full((*self.frame_dimensions[::-1], 3), mask, dtype=np.uint8)
            return frame_to_filter * filter_matrix

        if video_filter == VideoFilter.MONOCHROME:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif video_filter == VideoFilter.VALUE_INVERT:
            frame = 255 - frame

        elif video_filter == VideoFilter.RED_FILTER:
            filter_mask = np.array([0, 0, 1], dtype=np.uint8)
            frame = apply_filter_mask(filter_mask, frame)
        elif video_filter == VideoFilter.GREEN_FILTER:
            filter_mask = np.array([0, 1, 0], dtype=np.uint8)
            frame = apply_filter_mask(filter_mask, frame)
        elif video_filter == VideoFilter.BLUE_FILTER:
            filter_mask = np.array([1, 0, 0], dtype=np.uint8)
            frame = apply_filter_mask(filter_mask, frame)
        elif video_filter == VideoFilter.YELLOW_FILTER:
            filter_mask = np.array([0, 1, 1], dtype=np.uint8)
            frame = apply_filter_mask(filter_mask, frame)
        elif video_filter == VideoFilter.CYAN_FILTER:
            filter_mask = np.array([1, 1, 0], dtype=np.uint8)
            frame = apply_filter_mask(filter_mask, frame)
        elif video_filter == VideoFilter.MAGENTA_FILTER:
            filter_mask = np.array([1, 0, 1], dtype=np.uint8)
            frame = apply_filter_mask(filter_mask, frame)

        elif video_filter == VideoFilter.SWAP_RED_GREEN:
            frame[:, :, [1, 2]] = frame[:, :, [2, 1]]
        elif video_filter == VideoFilter.SWAP_GREEN_BLUE:
            frame[:, :, [0, 1]] = frame[:, :, [1, 0]]
        elif video_filter == VideoFilter.SWAP_BLUE_RED:
            frame[:, :, [0, 2]] = frame[:, :, [2, 0]]
        elif video_filter == VideoFilter.CYCLE_BLUE_GREEN_RED_ONCE:
            frame[:, :, [0, 1]] = frame[:, :, [1, 0]]
            frame[:, :, [1, 2]] = frame[:, :, [2, 1]]
        elif video_filter == VideoFilter.CYCLE_BLUE_GREEN_RED_TWICE:
            frame[:, :, [0, 1]] = frame[:, :, [1, 0]]
            frame[:, :, [0, 2]] = frame[:, :, [2, 0]]

        elif video_filter == VideoFilter.BGR_TO_HSV:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        elif video_filter == VideoFilter.RGB_TO_HSV:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        elif video_filter == VideoFilter.HSV_TO_BGR:
            frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
        elif video_filter == VideoFilter.HSV_TO_RGB:
            frame = cv.cvtColor(frame, cv.COLOR_HSV2RGB)
        elif video_filter == VideoFilter.BGR_TO_HLS:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
        elif video_filter == VideoFilter.RGB_TO_HLS:
            frame = cv.cvtColor(frame, cv.COLOR_HLS2BGR)
        elif video_filter == VideoFilter.HLS_TO_BGR:
            frame = cv.cvtColor(frame, cv.COLOR_HLS2BGR)
        elif video_filter == VideoFilter.HLS_TO_RGB:
            frame = cv.cvtColor(frame, cv.COLOR_HLS2RGB)
        elif video_filter == VideoFilter.BGR_TO_LAB:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        elif video_filter == VideoFilter.RGB_TO_LAB:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2LAB)
        elif video_filter == VideoFilter.LAB_TO_BGR:
            frame = cv.cvtColor(frame, cv.COLOR_LAB2BGR)
        elif video_filter == VideoFilter.LAB_TO_RGB:
            frame = cv.cvtColor(frame, cv.COLOR_LAB2RGB)
        elif video_filter == VideoFilter.BGR_TO_LUV:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2LUV)
        elif video_filter == VideoFilter.RGB_TO_LUV:
            frame = cv.cvtColor(frame, cv.COLOR_RGB2LUV)
        elif video_filter == VideoFilter.LUV_TO_BGR:
            frame = cv.cvtColor(frame, cv.COLOR_LUV2BGR)
        elif video_filter == VideoFilter.LUV_TO_RGB:
            frame = cv.cvtColor(frame, cv.COLOR_LUV2RGB)
        return frame
        
    def filter_resize_display_frame(self, frame, resize_factor=(1.0, 1.0), video_filter=VideoFilter.NO_FILTER):
        # Apply the filter to the video frame
        if video_filter != VideoFilter.NO_FILTER:
            frame = self.filter_frame(copy.deepcopy(frame), video_filter)
        # Resize video frame
        if resize_factor != (1.0, 1.0):
            frame = cv.resize(frame, np.multiply(resize_factor, self.frame_dimensions).astype(int))
        # Display the frame
        cv.imshow('', frame)
        return frame

    def save_static_video_stats(self, video_capture):
        if not video_capture.isOpened():
            return

        self.fps = video_capture.get(cv.CAP_PROP_FPS)
        self.frame_count = video_capture.get(cv.CAP_PROP_FRAME_COUNT)
        self.frame_count_digits = math.floor(math.log(self.frame_count, 10)) + 1
        self.frame_time_ms = int(1000.0 / self.fps)
        self.total_time_ms = 1000.0 * self.frame_count / self.fps
        self.frame_dimensions = (int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                                 int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))

    def open_recorder(self):
        time_string = get_time_string_from_seconds(time.time()).replace(':', '-').replace('.', ',')
        recording_filename = f"recording_{time_string}.mp4"
        self.recording_filepath = os.path.join(self.recording_directory, recording_filename)

        self.recording_capture = cv.VideoWriter(self.recording_filepath, cv.VideoWriter_fourcc(*'mp4v'),
                                                self.fps, self.recording_dimensions)
        print(f"Recording video to {self.recording_filepath}...")

    def release_recorder(self):
        self.recording_capture.release()
        print(f"Video written to {self.recording_filepath}.")
        pass

    # Play a video file.
    def play_video(self, resize_factor=(1.0, 1.0), start_time=0, duration=0, video_filter=None):
        frame = None
        jump_time = None
        jump_time_string = ''
        force_redisplay = False
        force_refresh = True
        self.user_request = UserRequest.NONE

        if self.video_filter_mode == VideoFilterMode.RANDOM:
            video_filter = random.randint(0, VideoFilter.FILTER_COUNT - 1)
        elif video_filter is None:
            video_filter = VideoFilter.NO_FILTER

        video_capture = cv.VideoCapture(self.video_file)
        self.save_static_video_stats(video_capture)

        # If in cutup mode, generate a random starting point with enough time to play
        # the full interval. If the video is too short, abort.
        if self.cutup_mode:
            if self.total_time_ms < self.cutup_interval:
                video_capture.release()
                return
            start_time = random.random() * (self.total_time_ms - self.cutup_interval)
            duration = self.cutup_interval
            print(f"Cutup mode on. Interval = {self.cutup_interval / 1000.0:0.3}s.")

        # Seek to the given start time
        if start_time > 0:
            video_capture.set(cv.CAP_PROP_POS_MSEC, start_time)
        # Save the current video time in milliseconds
        self.current_time_ms = video_capture.get(cv.CAP_PROP_POS_MSEC)

        self.print_basic_video_properties(video_capture)
        if video_filter != VideoFilter.NO_FILTER:
            print(f"Filter set to {VideoFilter(video_filter).name.replace('_', ' ')}")

        if self.is_recording and self.recording_capture is None:
            self.open_recorder()

        # Calculate end time
        end_time = self.total_time_ms if duration <= 0 else min(start_time + duration, self.total_time_ms)

        while True:
            # In the case that we're paused, and a filter has changed force a redisplay of the current frame.
            if force_redisplay:
                self.filter_resize_display_frame(frame, resize_factor, video_filter)
                force_redisplay = False
            # If unpaused, or we seek to a new time while paused, grab the next frame to display here
            elif not self.is_paused or force_refresh:
                # Get current video frame
                ret, frame = video_capture.read()
                force_refresh = False

                # Save the current video time in milliseconds
                self.current_time_ms = video_capture.get(cv.CAP_PROP_POS_MSEC)

            # Handle video EOF
            if not ret:
                if self.loop_mode:
                    # Seek back to the start of video
                    video_capture.set(cv.CAP_PROP_POS_MSEC, 0)
                    ret, frame = video_capture.read()
                else:
                    # End of video
                    break

            # Filter, resize, display frame
            filtered_frame = self.filter_resize_display_frame(frame, resize_factor, video_filter)

            if self.is_recording and not self.is_paused:
                # Resize video frame
                if self.recording_dimensions != self.frame_dimensions:
                    recorded_frame = cv.resize(filtered_frame, self.recording_dimensions)
                else:
                    recorded_frame = filtered_frame
                self.recording_capture.write(recorded_frame)

            # End playback if we hit the end time
            if self.current_time_ms >= end_time:
                break

            # Handle user input
            wait_key = cv.waitKey(self.frame_time_ms) & 0xFF
            wait_key_chr = chr(wait_key)

            # Press 'q' exit.
            if wait_key_chr == 'q':
                self.user_request = UserRequest.ABORT
                print("Aborting.")
                break

            #  Press 'n' for next video.
            if wait_key_chr == 'n':
                print("Next video...")
                break

            # Press 'h' for help.
            elif wait_key_chr == 'h':
                display_user_manual()

            # Press 'l' to toggle loop mode
            elif wait_key_chr == 'l':
                self.loop_mode = not self.loop_mode
                print(f"Loop mode {'on' if self.loop_mode else 'off'}.")

            # Press 'g' to toggle filter mode (normal, random)
            elif wait_key_chr == 'g':
                self.video_filter_mode = (self.video_filter_mode + 1) % VideoFilterMode.FILTER_MODES_COUNT
                print(f"Video filter mode set to {VideoFilterMode(self.video_filter_mode).name.replace('_', ' ')}")

            # Press backspace on Windows / delete on mac to jump to start of video
            elif wait_key_chr == '\b' or wait_key_chr == '\x7f':
                # Restart the video
                video_capture.set(cv.CAP_PROP_POS_MSEC, 0)
                if self.is_paused:
                    force_refresh = True
                print("Restarting video.")

            # Press 't' to set a time to jump back to with 'j'
            elif wait_key_chr == 't':
                jump_time = self.current_time_ms
                jump_time_string = get_time_string_from_seconds(jump_time / 1000.0)
                print(f"Jump time set to {jump_time_string}")

            # Press 'j' to jump back to the time set by 't'
            elif wait_key_chr == 'j':
                if jump_time is not None:
                    # Return to jump point
                    video_capture.set(cv.CAP_PROP_POS_MSEC, jump_time)
                    if self.is_paused:
                        force_refresh = True
                    print(f"Jumping back to {jump_time_string}")

            # Press space to pause
            elif wait_key_chr == ' ':
                self.is_paused = not self.is_paused
                print(f"Playback {'' if self.is_paused else 'un'}paused.")

            # Press 'r' to rewind 1 second, scaled by the current speed multiplier.
            # Press 'R' to rewind 5 seconds, scaled by the current speed multiplier.
            # Press 'f'/'F' similarly to fast-forward.
            elif wait_key_chr == 'f' or wait_key_chr == 'F' or wait_key_chr == 'r' or wait_key_chr == 'R':
                # Calculate seek interval. Seeking is proportional to the current playback rate. Seek will always
                # increment by 1 or 5 seconds as measured at the current playback speed. For example, if the
                # playback rate is 0.25x, a short seek will move forward 0.25 seconds as measured at a playback
                # rate of 1.0x., which is 1.0 s at the rate of 0.25x.
                speed_multiplier = self.speed_factors[self.speed_factor_index]
                if wait_key_chr == 'f':
                    seek_time = 1.0 * speed_multiplier
                elif wait_key_chr == 'F':
                    seek_time = 5.0 * speed_multiplier
                elif wait_key_chr == 'r':
                    seek_time = -1.0 * speed_multiplier
                else:
                    seek_time = -5.0 * speed_multiplier

                # Calculate target position in ms
                target_position_ms = self.current_time_ms + 1000.0 * seek_time
                # Constrain within the bounds of the video duration
                target_position_ms = min(max(0, target_position_ms), self.total_time_ms)
                frame_to_seek_time_string = get_time_string_from_seconds(target_position_ms / 1000.0)

                video_capture.set(cv.CAP_PROP_POS_MSEC, target_position_ms)
                if self.is_paused:
                    force_refresh = True
                print(f"Seeking {'+' if seek_time > 0 else ''}{seek_time}s to {frame_to_seek_time_string}.")

            # Press 's' to speed up or 'd' to slow down in a cycle of
            # (200%, 400%, 800%, 1600%, 6.25%, 12.5%, 25%, 50%, 100%)
            # (2x, 4x, 8x, 16x, 0.0625x, 0.125x, 0.25x, 0.5x, 1x)
            # Videos start at 100% speed by default
            elif wait_key_chr == 's' or wait_key_chr == 'd':
                speed_factor_increment = 1 if wait_key_chr == 'd' else -1
                self.speed_factor_index = (self.speed_factor_index + speed_factor_increment) % len(self.speed_factors)
                speed_multiplier = self.speed_factors[self.speed_factor_index]

                effective_fps = speed_multiplier * self.fps
                self.frame_time_ms = int(1000.0 / effective_fps)
                print(f"Setting speed to {speed_multiplier:0.3f}x = {effective_fps:0.3f} fps.")

            # Press 'a' to restore both speed up and slow down cycles to 100% and remove filter
            elif wait_key_chr == 'a':
                self.speed_factor_index = 0
                self.frame_time_ms = int(1000.0 / self.fps)
                video_filter = VideoFilter.NO_FILTER
                if self.is_paused:
                    force_redisplay = True
                print(f"Restoring normal speed 1.0x = {self.fps:0.3f} fps and removing filter.")

            # Press 'y' to increment filter. Press 'u' to decrement filter.
            elif wait_key_chr == 'y' or wait_key_chr == 'u':
                filter_increment = 1 if wait_key_chr == 'y' else -1
                video_filter = (video_filter + filter_increment) % VideoFilter.FILTER_COUNT
                if self.is_paused:
                    force_redisplay = True
                print(f"Filter set to {VideoFilter(video_filter).name.replace('_', ' ')}")

            # Press 'c' to toggle cutup mode.
            elif wait_key_chr == 'c':
                self.cutup_mode = not self.cutup_mode
                print(f"Cutup mode {'on' if self.cutup_mode else 'off'}.")

            # Press 'k' to set cutup interval interactively (in ms)
            elif wait_key_chr == 'k':
                # Create an instance of Tkinter frame
                user_window = tk.Tk()

                # Set the geometry of Tkinter frame
                user_window.geometry("250x100")
                user_window.resizable(False, False)

                # Initialize a Label to display the User Input
                label = tk.Label(user_window, text="Set cutup interval in ms", font=("Arial 14"))
                label.pack(pady=10)

                def set_cutup_interval():
                    interval = entry.get()
                    if not interval.isnumeric():
                        user_window.destroy()
                        return

                    interval = int(interval)
                    if interval <= 0:
                        user_window.destroy()
                        return

                    self.cutup_interval = interval
                    user_window.destroy()
                    print(f"Setting cutup interval to {interval}ms.")

                def handle_return_press(event):
                    set_cutup_interval()

                # Create an Entry widget to accept User Input
                entry = tk.Entry(user_window, width=12)
                entry.bind('<Return>', handle_return_press)
                entry.pack()
                # This only seems to work the first time
                entry.focus_set()

                user_window.mainloop()

            # Press 'i' to print basic video information.
            elif wait_key_chr == 'i':
                self.print_basic_video_properties(video_capture)

            # Press 'z' to toggle recording the video stream to a file.
            elif wait_key_chr == 'z':
                if self.is_recording:
                    self.release_recorder()
                else:
                    if self.recording_dimensions is None:
                        self.recording_dimensions = tuple(np.multiply(resize_factor,
                                                                      self.frame_dimensions).astype(int))
                    self.open_recorder()
                self.is_recording = not self.is_recording

            # Press 'o' to open a new custom file to play immediately
            elif wait_key_chr == 'o':
                # TO DO: Figure out why tkinter filedialog functions cause a
                # segmentation fault only on mac. For now, only allow loading
                # videos at startup on mac.
                if sys.platform == "win32":
                    root = tk.Tk()
                    root.withdraw()
                    video_file = filedialog.askopenfilename()

                    self.video_file = video_file
                    self.video_files_dict[video_file] = 0
                    self.user_request = UserRequest.FORCE_PLAY
                    print(f"Added {video_file} to video files.")
                    break
                else:
                    print("This feature doesn't work on mac. Looking into it...")

            # Press 'O' to select a directory of videos to add to the current video file collection
            elif wait_key_chr == 'O':
                # TO DO: Figure out why tkinter filedialog functions cause a
                # segmentation fault only on mac. For now, only allow loading
                # videos at startup on mac.
                if sys.platform == "win32":
                    self.load_videos_interactive(from_directory=True)
                else:
                    print("This feature doesn't work on mac. Looking into it...")

            # Press 'L' to select a individual video files to add to the current video file collection
            elif wait_key_chr == 'L':
                # TO DO: Figure out why tkinter filedialog functions cause a
                # segmentation fault only on mac. For now, only allow loading
                # videos at startup on mac.
                if sys.platform == "win32":
                    self.load_videos_interactive(from_directory=False)
                else:
                    print("This feature doesn't work on mac. Looking into it...")

            # Press backtick to print list of video files (debug feature)
            elif wait_key_chr == '`':
                self.print_video_file_list()

            # Handle any other key press as setting/recalling a favorite video
            elif wait_key != 255:
                # print(f"Wait key code: {wait_key}")
                wait_key_chr = chr(wait_key)

                # Clear the favorites
                if wait_key_chr == 'x':
                    self.favorites_map.clear()
                    print(f"Clearing favorites.")
                elif wait_key in self.favorites_map.keys():
                    # Cue up the favorite video if we're not already playing it
                    if self.favorites_map[wait_key] == self.video_file:
                        continue
                    self.video_file = self.favorites_map[wait_key]
                    self.user_request = UserRequest.FORCE_PLAY
                    print(f"Recalling favorite {wait_key_chr}")
                    break
                else:
                    already_mapped = False
                    for key in self.favorites_map.keys():
                        if self.video_file == self.favorites_map[key]:
                            already_mapped = True
                            print(f"This video is already saved to key {chr(key)} (code: {key})")
                            break
                    if not already_mapped:
                        self.favorites_map[wait_key] = self.video_file
                        print(f"Saving video to favorite {wait_key_chr} (code: {wait_key})")

        video_capture.release()

    def play_videos(self, with_replacement=False, resize_factor=(1.0, 1.0), cutup_mode=False, cutup_interval=1000,
                    video_filter=VideoFilter.NO_FILTER, video_filter_mode=VideoFilterMode.NORMAL,
                    recording_directory=None, recording_dimensions=None, record=False):
        file_count = self.video_file_count()
        if file_count == 0:
            print(f"Didn't find any video files. Load video files first.")
            return
        print(f"Found {file_count} video{'s' if file_count != 1 else ''}.")

        self.cutup_mode = cutup_mode
        self.cutup_interval = cutup_interval
        self.video_filter_mode = video_filter_mode

        if recording_directory is not None:
            self.recording_directory = recording_directory
        if recording_dimensions is not None:
            self.recording_dimensions = recording_dimensions

        if record:
            self.is_recording = True

        # If random with replacement, set up iterator to a pick random file from video files. Calling next() on this
        # iterator will always return a video file. If random without replacement, set up an empty iterator to be
        # initialized below, as it follows the same pattern as when we've exhausted the iterator.
        video_files = list(self.video_files_dict.keys())
        video_iterator = iter(lambda: random.choice(video_files), None) if with_replacement else iter([])

        while self.user_request != UserRequest.ABORT:
            try:
                # This will only fail if there's nothing left to iterate over, which is only in the without replacement
                # case. When we have run out of unique video files, as well as initially when we have an empty iterator,
                # this call to next() will raise a StopIteration exception.
                self.video_file = next(video_iterator)

                video_file_done = False
                
                # Play the video. When the user requests a favorite video, the current video will get unloaded,
                # the player's video file will be redirected to the favorite video, and the user request state will
                # indicate REPLAY.
                # If play_video() returns False, we can move on to the next random video.
                while self.user_request == UserRequest.FORCE_PLAY or not video_file_done:
                    self.play_video(resize_factor=resize_factor, video_filter=video_filter)
                    video_file_done = True

            except StopIteration:
                # The iterator is empty. Randomize the file order, re-initialize the iterator, and try again.
                video_files = list(self.video_files_dict.keys())
                random.shuffle(video_files)
                video_iterator = iter(video_files)

        if self.is_recording:
            self.recording_capture.release()
            print(f"Video written to {self.recording_filepath}.")
            self.is_recording = False

def run():
    video_player = VideoPlayer()
    video_player.load_videos_interactive()
    video_player.play_videos()

run()







