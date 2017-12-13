import cv2
import pyautogui
import os
from mss import mss
import numpy as np


class GameBoard(object):

    def __init__(self, img_threshold=.6):
        """
        :param img_threshold: Confidence in that you find a match image
        """
        self.img_threshold = img_threshold
        self.roi = None
        self.found_roi = False
        self.__find_roi()
        self.empty_obstacles = [600, 0]

    def get_game_img(self):
        """
        :return: A grey-scale image of the region of interest (or the whole screen if no roi was passed)
        """
        with mss() as sct:
            if self.roi is None:
                filename = sct.shot(output='Images/fullscreen.PNG')
                screen_shot = cv2.imread(filename)
            else:
                screen_shot = sct.grab(self.roi)

        screen_shot = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGR2GRAY)

        return screen_shot

    def find_obstacles(self, game_img, max_gap=50):
        """
        Gets the obstacles in the frame.

        :param game_img: The gray image of the game board
        :param max_gap: Max amount of pixels allowed between obstacles before they are called two different obstacles
        :param none_distance: Distance used if no obstacle are found
        :param none_width: Width of obstacle if none are found
        :return: State of obstacles - [location obst 1, width obst 1, location obst 2, width obst 2, ...]
        """

        # Get the std, average pixel value for the whole game image
        avg = np.mean(game_img)
        std = np.std(game_img)

        # Get avg pixel value for each column in the image
        avg_col = np.mean(game_img, axis=0)

        # Find the columns that are blacker than average (by 1 std). This is where the obstacles are!
        obstacles = np.where(avg_col < avg - std)
        obstacles = list(obstacles[0])

        # If no obstacles are found, return the default none state
        if len(obstacles) < 1:
            return self.empty_obstacles

        # Currently the obstacle list holds the column of the obstacles. This array can vary in length
        # which can be annoying to deal with. This block converts obstacle array
        # from something like: [13,14,15,16,20,21,105,106,107,130,131,133]
        # To: [13, 18, 105, 2, 130, 3] or ([obst1, width1, obst2, width2, obst3, width3])
        cleaned_obstacles = []
        start_obs = obstacles[0]
        for pixel, next_pixel in zip(obstacles, obstacles[1:]):
            if next_pixel - pixel > max_gap:
                cleaned_obstacles.append(start_obs)
                cleaned_obstacles.append(pixel - start_obs + 1)
                start_obs = next_pixel

        cleaned_obstacles.append(start_obs)
        cleaned_obstacles.append(obstacles[-1] - start_obs + 1)

        # Only keep 1st obstacle
        if len(cleaned_obstacles) > 2:
            return cleaned_obstacles[:2]

        return cleaned_obstacles

    def game_over(self, game_img):
        """
        Finds if the game is over

        :param game_img: Screen shot of game
        Passed into cv2.matchTemplate
        :return: Boolean if the game ended
        """
        game_over_img = cv2.imread(os.path.join('Images', 'game_over.PNG'), 0)
        res = cv2.matchTemplate(game_img, game_over_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        return max_val > self.img_threshold

    def go_to_roi(self):
        pyautogui.moveTo(self.roi['left'] + self.roi['width']/2,
                         self.roi['top'] + self.roi['height']/2)
        pyautogui.click()

    def __find_roi(self):
        """
        Gets the region of interest (the game) in the screenshot.

        :return: top, left, width, and height of the region of interest
        """

        dino_info = self.__find_dino()
        if not dino_info:
            return None

        # get's the width and height of the game board
        game_board = cv2.imread(os.path.join('Images', 'roi.PNG'), 0)
        h, w = game_board.shape

        roi = {
            'top': dino_info['top_left'][1] - dino_info['h'],
            'left': int(dino_info['top_left'][0] + dino_info['w']*1.5),  # add 50% onto the width (dino moves at start)
            'width': w - dino_info['w'],
            'height': 2 * dino_info['h']
        }
        self.roi = roi
        self.found_roi = True

    def __find_dino(self):
        """
        The dino is a good image to use as a template to find the region of interest. This method
        takes a screenshot, and looks for the dino.

        :return: The location of the dino in the screenshot.
        """

        dino = cv2.imread(os.path.join('Images', 'dino.PNG'), 0)
        h, w = dino.shape

        screen_shot = self.get_game_img()
        res = cv2.matchTemplate(screen_shot, dino, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # return none if we can't find the dino within the threshold
        if max_val < self.img_threshold:
            return None

        top_left = max_loc
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        return {'top_left': top_left, 'bottom_right': bottom_right, 'h': h, 'w': w}
