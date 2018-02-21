'''
DataParser.py
Assignment #3
'''
# Returns list of dictonaries, each dictionary is data associated with a heatmiser

from BinHelper import createBins
from BinHelper import createBinsManually

# Returns list of dictonaries, each dictionary is data associated with a heatmiser
def parseDataJSON(fname):
    with open(fname, "r") as f:
        f.readline()  # skip header
        # Iterate line by line
        data = []
        headers = ['ID', 'Distance', 'Speeding', 'Location', 'OSHA']
        for line in f:
            line = line.strip()

            # Redo tabs with spaces
            line = line.replace('\t', ',')

            lineList = line.split(",")

            heatmiser = {}
            for i in range(len(lineList)):
                heatmiser[headers[i]] = lineList[i]

            data.append(heatmiser)
    return data

# Returns list of lists, each list is data associated with a heatmiser
def parseDataList(fname):
    with open(fname, "r") as f:
        f.readline()  # skip header
        # Iterate line by line
        data = []
        headers = ['ID', 'Distance', 'Speeding', 'Location', 'OSHA']
        for line in f:
            line = line.strip()

            # Redo tabs with spaces
            line = line.replace('\t', ',')

            heatmiser = line.split(",")
            data.append(heatmiser)
    return data


def parseDataArrays(fname):
    with open(fname, "r") as f:
        f.readline()  # skip header
        # Iterate line by line
        data = {}
        ids = []
        distances = []
        speeds = []
        locations= []
        oshas = []
        headers = ['ID', 'Distance', 'Speeding', 'Location', 'OSHA']

        for line in f:
            line = line.strip()
            # Redo tabs with spaces
            line = line.replace('\t', ',')

            lineList = line.split(",")

            ids.append(lineList[0])
            distances.append(float(lineList[1]))
            speeds.append(float(lineList[2]))
            locations.append(lineList[3])
            oshas.append(lineList[4])

    # transform numerical values into categorical values
    # speeds = createBins(speeds, "speed", [], 5)
    # distances = createBins(distances, "distance", [], 5)
    distances = createBinsManually(distances, "distance")
    speeds = createBinsManually(speeds, "speed")

    data[headers[0]] = ids
    data[headers[1]] = distances #.tolist()
    data[headers[2]] = speeds #.tolist()
    data[headers[3]] = locations
    data[headers[4]] = oshas
    return data

def getDataArrays():
    fname = 'HW3_Data.txt'
    data = parseDataArrays(fname)
    return data

def getDataJSON():
    fname = 'HW3_Data.txt'
    data = parseDataJSON(fname)
    return data

def getDataList():
    fname = 'HW3_Data.txt'
    data = parseDataList(fname)
    return data

def main():
    fname = 'HW3_Data.txt'
    # data = parseDataList(fname)
    # data = parseDataArrays(fname)


if __name__ == '__main__':
    main()
