from selenium import webdriver  # Importing necessary modules from selenium
from selenium.webdriver.common.by import By  # For locating elements
from selenium.webdriver.chrome.service import Service  # For managing the ChromeDriver service
from selenium.webdriver.chrome.options import Options  # For setting Chrome options
from selenium.webdriver.common.keys import Keys  # For sending keyboard actions
from PIL import Image  # For image manipulation
from io import BytesIO  # For handling binary I/O
from webdriver_manager.chrome import ChromeDriverManager  # For automatic ChromeDriver management
import time  # For adding delays
import base64  # For encoding and decoding base64
import numpy as np  # For numerical operations on arrays
import cv2  # For image processing
import copy  # For copying objects

class Game():
    def __init__(self, game_url, chrome_driver_path, init_script):
        # Initialize the game with URL, driver path, and initial script
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"
        
        # Setting up Chrome options for headless mode and other configurations
        chrome_options = Options()
        #chrome_options.add_argument('headless')  # Not showing browser is faster
        #chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        
        # Setting up the ChromeDriver service
        service = Service(ChromeDriverManager().install()) 
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Positioning and sizing the browser window
        self.driver.set_window_position(x=-10, y=0)
        print(self.driver.get_window_size())  # Print the size of browser
        self.driver.set_window_size(1450, 1080)
        self.driver.get(game_url)
        self.driver.execute_script("Runner.config.ACCELERATION=0")  # Disable acceleration for easy mode (no birds)
        time.sleep(1)  # Wait for the HTML to load
        self.driver.execute_script(init_script)  # Set the ID for the canvas element
        self.CV_display = self.show_img()  # Show the game state using OpenCV
        self.CV_display.__next__()  # Initialize the display coroutine
    
    def screen_shot(self):
        # Take a screenshot of the game
        image_b64 = self.driver.execute_script(self.getbase64Script)
        np_img = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        self.canvas_image = copy.deepcopy(np_img)
        np_img = cv2.resize(np_img, (80, 80))  
        return np_img

    def show_img(self, graphs=False):
        # Display the game state using OpenCV
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "Gameplay"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            imS = cv2.resize(screen, (200, 130))  
            cv2.imshow(window_title, imS)
            if (cv2.waitKey(1) & 0xFF == ord('q')): # Close window on 'q' key press
                cv2.destroyAllWindows()
                break
    
    def save_gif(self):
        # Placeholder for saving gameplay as GIF
        pass
            
    def get_state(self, actions, grayscale_cam=None):
        # Retrieve the current state of the game
        reward = 0.1
        is_over = False  # Game over
        if actions[1] == 1:
            self.press_up()
        #elif actions[1] == 2:
        #    self.press_down()
        image = self.screen_shot()
        self.CV_display.send(image)
        
        if self.get_crashed(): # Check if the game is over
            reward = -1
            is_over = True
        
        return image, reward, is_over  # Return the state tuple

    def get_crashed(self):
        # Check if the game has crashed
        return self.driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        # Check if the game is currently playing
        return self.driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        # Restart the game
        self.driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        # Simulate pressing the up arrow key
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        # Simulate pressing the down arrow key
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        # Retrieve the current score from the game
        score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)  # The javascript object is of type array with score in the format [1,0,0] which is 100.
        return int(score)

    def pause(self):
        # Pause the game
        return self.driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        # Resume the game
        return self.driver.execute_script("return Runner.instance_.play()")

    def end(self):
        # Close the browser and end the game session
        self.driver.close()