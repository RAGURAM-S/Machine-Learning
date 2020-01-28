import pandas
dataset = pandas.read_csv("C:/ml/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/Market_Basket_Optimisation.csv", header = None)

transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

from apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

result = list(rules)