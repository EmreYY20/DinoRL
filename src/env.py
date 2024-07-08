############ import libraries ############

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
from webdriver_manager.chrome import ChromeDriverManager
import time
import base64
import numpy as np
import cv2
import copy

#########################################

class Game():
    def __init__(self, game_url, chrome_driver_path, init_script):
        # initialize the game with URL, driver path, and initial script
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"
        
        # setting up Chrome options for headless mode and other configurations
        chrome_options = Options()
        #chrome_options.add_argument('headless')  # not showing browser is faster
        #chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        
        # setting up the ChromeDriver service
        service = Service(ChromeDriverManager().install()) 
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # positioning and sizing the browser window
        self.driver.set_window_position(x=-10, y=0)
        print(self.driver.get_window_size())  # print the size of browser
        self.driver.set_window_size(1450, 1080)
        self.driver.get(game_url)
        self.driver.execute_script("Runner.config.ACCELERATION=0")  # disable acceleration for easy mode (no birds)
        time.sleep(1)  # wait for the HTML to load
        self.driver.execute_script(init_script)  # set the ID for the canvas element
        self.CV_display = self.show_img()  # show the game state using OpenCV
        self.CV_display.__next__()  # initialize the display coroutine
    
    def screen_shot(self):
        # take a screenshot of the game
        image_b64 = self.driver.execute_script(self.getbase64Script)
        np_img = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        self.canvas_image = copy.deepcopy(np_img)
        np_img = cv2.resize(np_img, (80, 80))  
        return np_img

    def show_img(self, graphs=False):
        # display the game state using OpenCV
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "Gameplay"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            imS = cv2.resize(screen, (200, 130))  
            cv2.imshow(window_title, imS)
            if (cv2.waitKey(1) & 0xFF == ord('q')): # close window on 'q' key press
                cv2.destroyAllWindows()
                break
    
    def save_gif(self):
        # placeholder for saving gameplay as GIF
        pass
            
    def get_state(self, actions, grayscale_cam=None):
        # retrieve the current state of the game
        reward = 0.1
        is_over = False  # game over
        if actions[1] == 1:
            self.press_up()
        #elif actions[1] == 2:
        #    self.press_down()
        image = self.screen_shot()
        self.CV_display.send(image)
        
        if self.get_crashed(): # check if the game is over
            reward = -1
            is_over = True
        
        return image, reward, is_over  # return the state tuple

    def get_crashed(self):
        # check if the game has crashed
        return self.driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        # check if the game is currently playing
        return self.driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        # restart the game
        self.driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        # simulate pressing the up arrow key
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        # simulate pressing the down arrow key
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        # retrieve the current score from the game
        score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)  # the javascript object is of type array with score in the format [1,0,0] which is 100.
        return int(score)

    def pause(self):
        # pause the game
        return self.driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        # resume the game
        return self.driver.execute_script("return Runner.instance_.play()")

    def end(self):
        # close the browser and end the game session
        self.driver.close()
