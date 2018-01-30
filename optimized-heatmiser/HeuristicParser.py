'''
HeuristicParser.py
Assignment #2
'''


def parseHeuristic(fname):
	with open(fname, "r") as f:
		f.readline() # skip header
		# Iterate line by line
		data = {}
		for line in f:
			line = line.strip()

			start_char = int(line[0])
			end_node = None
			atHeuristic = False
			print(start_char)

			# Add dictionary entry for start node
			if start_char not in data:
				data[start_char] = {}

			# Iterate through line
			for i in range(1, len(line)):
				char = line[i]
				# Don't include spaces
				if (char != '\t') and (not atHeuristic):
					end_node = int(char)
					atHeuristic = True
				elif (char != '\t') and atHeuristic:
					data[start_char] = {end_node: int(char)}
					break
		print(data)
			
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