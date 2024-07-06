############ import libraries ############

import numpy as np 
import matplotlib.pyplot as plt  

#########################################

# load the scores from a NumPy file
scores = np.load('./test_scores/DoubleDQN_config2.npy')

# plot the scores with a label and some transparency
plt.plot(scores, label='score', alpha=0.7)

# add text annotations for mean, median, and minimum score
plt.text(85, np.mean(scores), 'mean (' + str(np.mean(scores)) +')')
plt.text(85, np.median(scores), 'median (' + str(np.median(scores)) + ')')
plt.text(85, np.min(scores), 'min (' + str(np.min(scores))+')')

# make x axis every episode, show
plt.xticks(np.arange(0, 20, 2))

# draw horizontal lines for mean, median, and minimum scores
plt.axhline(y=np.mean(scores), color='r', linestyle='--', label='mean')
plt.axhline(y=np.median(scores), color='g', linestyle='-.', label='median')
plt.axhline(y=np.min(scores), color='purple', linestyle=':', label='min')

# label the x-axis and y-axis
plt.xlabel('Episodes')
plt.ylabel('Scores')

# add a title to the plot
plt.title('Testing Scores for 20 Episodes (Config 2)')

# add a legend to the plot
plt.legend()

# save the plot as a PNG file
plt.savefig('./misc/plots/ddqn_testing_scores_config2.png')

# clear the current figure
plt.clf()
