'''
DataParser.py
Assignment #3
'''

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

            lineList = line.split(",")

            heatmiser = []
            # Only numbers for now
            # for i in range(len(lineList)):
            for i in range(1, 3):
                heatmiser.append(lineList[i])
            data.append(heatmiser)
    return data

def getDataList():
    fname = 'HW3_Data.txt'
    data = parseDataList(fname)
    return data

def getDataJSON():
    fname = 'HW3_Data.txt'
    data = parseDataJSON(fname)
    return data

def main():
    fname = 'HW3_Data.txt'
    data = parseDataList(fname)


if __name__ == '__main__':
    main()
