'''
HeuristicParser.py
Assignment #2
'''


def parseHeuristic(fname):
    with open(fname, "r") as f:
        f.readline()  # skip header
        # Iterate line by line
        data = {}
        for line in f:
            line = line.strip()

            line = line.replace('\t\t\t\t', ',')
            line = line.replace('\t\t\t', ',')
            line += ','

            if (line[1] == ","):
                start_int = int(line[0])
                start = 2

            else:
                start_int = int(line[0:2])
                start = 3

            end_node = None
            atHeuristic = False

            # Add dictionary entry for start node
            if start_int not in data:
                data[start_int] = {}

            # Iterate through line
            for i in range(start, len(line)):
                if (line[i] != ',' and line[i + 1] == ','):
                    if (i > 0 and line[i - 1] != ','):
                        char = line[i - 1:i + 1]
                    else:
                        char = line[i]

                    # Don't include spaces
                    if (char != '\t') and (not atHeuristic):
                        end_node = int(char)
                        atHeuristic = True
                    elif (char != '\t') and atHeuristic:
                        data[start_int][end_node] = int(char)
                        break

    return data


def getHeuristic():
    fname = 'HeatMiserHeuristic.txt'
    data = parseHeuristic(fname)
    return data


def main():
    fname = 'HeatMiserHeuristic.txt'
    data = parseHeuristic(fname)


if __name__ == '__main__':
    main()
