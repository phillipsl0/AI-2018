# from numpy import split, array
from pandas import cut, qcut

speedLabels = ["Very Slow", "Slow", "Neutral", "Fast", "Very Fast"]
distanceLabels = ["Very Close", "Close", "Neutral", "Far", "Very Far"]

def createBins(data, dataType, dataLabels, binNumber):
    if (dataType == "speed"):
        dataLabels = speedLabels
    elif (dataType == "distance"):
        dataLabels = distanceLabels

    # splits data into 5 approximately even bins
    result = cut(data, binNumber, labels=dataLabels)

    return result

def createBinsManually(data, dataType):
    newLabels = []

    for i in range(4000):
       if (dataType == "speed"):
        if (data[i] < 20):
            newLabels.append("Very Slow")
        elif (20 <= data[i] and data[i] < 40):
            newLabels.append("Slow")
        elif (40 <= data[i] and data[i] < 60):
            newLabels.append("Neutral")
        elif (60 <= data[i] and data[i] < 80):
            newLabels.append("Fast")
        elif (data[i] >= 80):
            newLabels.append("Very Fast")
       else:
           if (data[i] < 53.43):
               newLabels.append("Very Close")
           elif (53.43 <= data[i] and data[i] < 101.27):
               newLabels.append("Close")
           elif (101.27 <= data[i] and data[i] < 149.11):
               newLabels.append("Neutral")
           elif (149.11 <= data[i] and data[i] < 196.95):
               newLabels.append("Far")
           elif (data[i] >= 196.95):
               newLabels.append("Very Far")

    return newLabels

def main():
    test = [5, 3, 4, 6, 3, 10, 9, 12, 32, 2]
    dataLabels = ["Very Slow", "Slow", "Neutral", "Fast", "Very Fast"]
    createBins(test, "speed", dataLabels, 5)

if __name__ == '__main__':
    main()