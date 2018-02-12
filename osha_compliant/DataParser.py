'''
DataParser.py
Assignment #3
'''


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


def getData():
    fname = 'HW3_Data.txt'
    data = parseData(fname)
    return data


def main():
    fname = 'HW3_Data.txt'
    data = parseData(fname)


if __name__ == '__main__':
    main()
