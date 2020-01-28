import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/UCB/Ads_CTR_Optimisation.csv")

n = 10000
d = 10

ads_selected = []
total_rewards = 0

import random

for i in range(0, n):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[i, ad]
    total_rewards = total_rewards + reward
    
import matplotlib.pyplot as plotter
plotter.hist(ads_selected)
plotter.title('Histogram of selected ads')
plotter.xlabel('Ads')
plotter.ylabel('Number of clicks on the ad')
plotter.show()