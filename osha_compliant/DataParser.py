'''
DataParser.py
Assignment #3
'''


def parseData(fname):
    with open(fname, "r") as f:
        f.readline()  # skip header
        # Iterate line by line
        data = []
        for line in f:
            line = line.strip()

            # Redo tabs with spaces
            line = line.replace('\t', ',')

            print(line)
            line_list = line.split(",")
            print(line_list)

            # end_node = None
            # atHeuristic = False

            # # Add dictionary entry for start node
            # if start_int not in data:
            #     data[start_int] = {}

            # # Iterate through line
            # for i in range(start, len(line)):
            #     if (line[i] != ',' and line[i + 1] == ','):
            #         if (i > 0 and line[i - 1] != ','):
            #             char = line[i - 1:i + 1]
            #         else:
            #             char = line[i]

            #         # Don't include spaces
            #         if (char != '\t') and (not atHeuristic):
            #             end_node = int(char)
            #             atHeuristic = True
            #         elif (char != '\t') and atHeuristic:
            #             data[start_int][end_node] = int(char)
            #             break

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
