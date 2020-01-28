import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/UCB/Ads_CTR_Optimisation.csv")

import random

N = 10000
d = 10

number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
ads_selected = []
total_rewards = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    
    total_rewards = total_rewards + reward
    
#histogram
import matplotlib.pyplot as plotter
plotter.hist(ads_selected)
plotter.title('Histogram of Ads Selection')
plotter.xlabel('Different versions of Ads')
plotter.ylabel('Number of times each version of the Ads were selected')
plotter.show()

