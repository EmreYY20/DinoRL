from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

scores = np.load('./test_scores/DoubleDQN.npy')
#print(scores)
plt.plot(scores, label='score', alpha=0.7)
plt.text(85, np.mean(scores), 'mean (' + str(np.mean(scores)) +')')
plt.text(85, np.median(scores)-5000, 'median (' + str(np.median(scores)) + ')')
plt.text(85, np.min(scores)-5000, 'min (' + str(np.min(scores))+')')

plt.axhline(y=np.mean(scores), color='r', linestyle='--', label='mean')
plt.axhline(y=np.median(scores), color='g', linestyle='-.', label='median')
plt.axhline(y=np.min(scores), color='purple', linestyle=':', label='min')
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('Testing scores in 100 episodes')
plt.legend()
plt.savefig('./misc/plots/ddqn_testing_scores.png')
plt.clf()

