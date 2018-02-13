# from numpy import split, array
from pandas import qcut

speedLabels = ["Very Slow", "Slow", "Neutral", "Fast", "Very Fast"]
distanceLabels = ["Very Close", "Close", "Neutral", "Far", "Very Far"]

def createBins(data, dataType, dataLabels, binNumber):
    if (dataType == "speed"):
        dataLabels = speedLabels
    elif (dataType == "distance"):
        dataLabels = distanceLabels

    # splits data into 5 approximately even bins
    result = qcut(data, binNumber, labels=dataLabels)

    return result


def main():
    test = [5, 3, 4, 6, 3, 10, 9, 12, 32, 2]
    dataLabels = ["Very Slow", "Slow", "Neutral", "Fast", "Very Fast"]
    createBins(test, "speed", dataLabels, 5)

if __name__ == '__main__':
    main()