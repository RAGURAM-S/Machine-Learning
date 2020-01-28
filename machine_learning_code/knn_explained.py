import math

def classifyApoint(points, p,k):
    distance = []
    for key in points:
        for values in points[key]:
            euclidean_distance = math.sqrt((values[0]-p[0])**2 + (values[1]-p[1])**2)
            distance.append((euclidean_distance, key))
    distance = sorted(distance)[:k]

    frequency1 = 0
    frequency2 = 0
    
    for d in distance:
        if d[1] == 0:
            frequency1 = frequency1 + 1
        elif d[1] == 1:
            frequency2 = frequency2 + 1
    if frequency1 > frequency2:
        return 0
    else:
        return 1
    

points = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)], 
          1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]} 
  
p = (2.5,7)
k = 5
print("The value classified to unknown point is: \n {}".format(classifyApoint(points,p,k)))  

