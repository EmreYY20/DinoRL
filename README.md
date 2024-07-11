# Applying Reinforcement Learning for the Chorme Dino Run Game

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![Dino_non-birthday_version](https://github.com/EmreYY20/DinoRL/assets/120115560/9609942f-1c2a-403f-a98d-7f987b34fb54)


## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## About

A Reinforcement Learning project that trains various Double Deep Q-Networks (DDQN) to excel in the Chrome Dino Game by dodging obstacles and maximizing its score through iterative learning.

Team members:
- [Emre Iyigün](https://github.com/EmreYY20) 
- [Aref Hasan](https://github.com/aref-hasan) 
- [Nik Yakovlev](https://github.com/nikyak10)
- [Ilgar Korkmaz](https://github.com/ilgark)

#### Implementation of the DDQN
The diagram illustrates the architecture of the DDQN. The agent receives a stack of 4 images (80x80
pixels each) representing the last states and uses its Q-Network to predict Q-values for possible actions ("Do nothing"
or "Jump"). Based on these Q-values, the agent selects an action and interacts with the environment, resulting in a new
state and reward. 
![RL (4) (1)](https://github.com/EmreYY20/DinoRL/assets/120115560/5faf7020-1ad0-4afe-8773-e98f1855b7f1)

#### Training Results
![score_chart](https://github.com/EmreYY20/DinoRL/assets/120115560/0ef0b521-d203-40f7-9c27-8750f8d4fe19)
![reward_chart](https://github.com/EmreYY20/DinoRL/assets/120115560/8bc83af4-39e9-4675-b446-1cdcd1bbc34a)
![loss_chart](https://github.com/EmreYY20/DinoRL/assets/120115560/10225c7f-7f63-4378-9934-d34fe708bfd5)

#### Test Results
The chart compares the maximum scores reached by different variants in 20 test rounds. Variants 1
(Blue), 2 (Orange), and 3 (Green) also surpassed the baseline (Grey), with Variant 3 (Green) performing slightly better
than Variants 1 and 2.
![max](https://github.com/EmreYY20/DinoRL/assets/120115560/0889d72f-33a7-4151-a05e-0c7515c159f5)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Chrome-Browser
  
### Installation

1. Clone the repository:

   ```bash
   git clone EmreYY20/DinoRL

2. Navigate to the project directory:
   ```bash
   cd DinoRL

3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt

## Usage

1. Start localhost with:

   ```bash
   python -m http.server 8000
   ```

2. Run the code by selecting a config file. In the config file select if you want to train or test:

   ```bash
   python main.py -c config/config1
   ```

   To the run the baseline:
   ```bash
   python baseline/main_modified.py -c config/baseline_config
   ```
After training, a tfevents-file is created in the runs folder which can be openend using TensorBoard.
   

## License
This project is licensed under the MIT License - see the [License](LICENSE) file for details. 
