'''
DataParser.py
Assignment #3
'''

from BinHelper import createBins

def parseData(fname):
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

    print(data)
    return data


def parseDataArrays(fname):
    with open(fname, "r") as f:
        f.readline()  # skip header
        # Iterate line by line
        data = {}
        ids = []
        speeds = []
        distances = []
        locations= []
        oshas = []
        headers = ['ID', 'Distance', 'Speeding', 'Location', 'OSHA']

        for line in f:
            line = line.strip()

            # Redo tabs with spaces
            line = line.replace('\t', ',')

            lineList = line.split(",")

            ids.append(lineList[0])
            speeds.append(float(lineList[1]))
            distances.append(float(lineList[2]))
            locations.append(lineList[3])
            oshas.append(lineList[4])

    # transform numerical values into categorical values
    speeds = createBins(speeds, "speed", [], 5)
    distances = createBins(distances, "distance", [], 5)

    data[headers[0]] = ids
    data[headers[1]] = speeds
    data[headers[2]] = distances
    data[headers[3]] = locations
    data[headers[4]] = oshas

    return data

def getData():
    fname = 'HW3_Data.txt'
    data = parseData(fname)
    return data


def main():
    fname = 'HW3_Data.txt'
    data = parseDataArrays(fname)


if __name__ == '__main__':
    main()
