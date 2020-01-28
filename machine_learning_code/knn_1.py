file = open("C:/Users/1024982/Desktop/New Text Document.txt", 'r')
lines = file.read().splitlines()
file.close()

features = lines[0].split(', ')[:-1]

