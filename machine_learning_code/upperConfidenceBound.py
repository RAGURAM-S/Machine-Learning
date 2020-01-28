import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/UCB/Ads_CTR_Optimisation.csv")

import math

N = 10000
d = 10

ads_selected = []
number_of_selections = [0]*d
sum_of_rewards = [0]*d
total_rewards = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (number_of_selections[i] > 0):
            average_rewards = sum_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/(number_of_selections[i]))
            upper_bound = delta_i + average_rewards
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_rewards = total_rewards + reward
    
#histogram
import matplotlib.pyplot as plotter
plotter.hist(ads_selected)
plotter.title('Histogram of Ads Selection')
plotter.xlabel('Different Version of Ads')
plotter.ylabel('Number of times each version of the Ads were selected')
plotter.show()

