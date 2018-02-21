'''
OSHACompliantHeatMiser.py
Assignment #3
'''

import DataParser as dp
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import random

'''
Implementation of k-means clustering using distance, speed
Using a library: https://pythonprogramminglanguage.com/kmeans-elbow-method/
'''
class KMeansClustering:
	def __init__(self, k=2, maxIter=10, fileName="osha_heatmiser_trial_output_kmeans"):
		self.k = k  # number of clusters
		self.centroids = {}
		self.classification = {}  # dictionary containing key of a dist, speed point and value of cluster index it belongs to
		self.maxIter = maxIter
		self.hasUpdated = True
		self.oshaStatus = {}
		self.locationStatus = {}
		self.fileName = (fileName + str(k))

	# Converts heatmiser arrays into a numpy array of distance, speed values
	# Gets heatmiser status.
	def processData(self, data):
		arr = []
		for heatmiser in data:
			point = [float(heatmiser[1]), float(heatmiser[2])]
			arr.append(point)
			self.oshaStatus[self.pointToKey(point)] = heatmiser[4]  # Get OSHA status for heatmiser
			self.locationStatus[self.pointToKey(point)] = heatmiser[3]  # Get location type for heatmiser
		return arr

	def calcEuclideanDistance(self, p, q):
		pX = p[0]
		pY = p[1]

		qX = q[0]
		qY = q[1]

		return math.sqrt(((qX - pX) ** 2) + ((qY - pY) ** 2))

	# Initialize clusters based on points furthest away in data
	# ONLY WORKS FOR K = 2
	def initClustersFurthest(self, points):
		maxDist = 0
		point = None

		# Iterate through points
		for h in range(len(points)):
			j = points[h]
			for i in range(len(points)):
				# Skip if same point
				k = points[i]
				if i == h:
					continue

				dist = self.calcEuclideanDistance(j, k)

				# Found point further away
				if dist > maxDist:
					maxDist = dist

					# Assign to centroids
					self.centroids[0] = j
					self.classification[self.pointToKey(j)] = 0

					self.centroids[1] = k
					self.classification[self.pointToKey(k)] = 1
					self.k = 2

	# Initialize clusters randomly
	def initClustersRandom(self, points):
		# Randomly choose k points from data
		randPoints = random.sample(points, self.k)

		# Initialize cluster centers as points
		for i in range(self.k):
			point = randPoints[i]
			self.centroids[i] = point
			self.classification[self.pointToKey(point)] = i

	# Helper function that converts point to string
	def pointToKey(self, point):
		strPoint = [str(val) for val in point]
		key = ','.join(strPoint)
		return key

	# Helper function that converts string to array
	# Ex. "9.0,4.57" --> [9.0, 4.57]
	def keyToPoint(self, key):
		strPoint = key.split(",")
		arr = [float(strPoint[0]), float(strPoint[1])]
		return arr

	# Update average of given centroid
	def updateClusterCentroid(self, index):
		total = 0
		sumX = 0
		sumY = 0
		for heatmiser in self.classification:
			if self.classification[heatmiser] == index:
				sumX += self.keyToPoint(heatmiser)[0]
				sumY += self.keyToPoint(heatmiser)[1]
				total += 1

		newX = sumX / total
		newY = sumY / total
		self.centroids[index] = [newX, newY]

	# Fit points to appropriate clusters
	def fit(self, points, first):
		# Assign training instance to cluster with closest centroid
		count = 0
		self.hasUpdated = False

		for heatmiser in points:
			count += 1
			# Print to one line dynamically
			sys.stdout.write("Fitting point " + str(count) + " out of " + str(
				len(points)) + "                                                \r", )
			sys.stdout.flush()

			key = self.pointToKey(heatmiser)
			# If initializing, assign point to nearest cluster
			if first:
				self.hasUpdated = True  # indicates update will happen
				closestCluster = None
				minDist = float('inf')

				# Check distance to each centroid
				for centroidIndex in range(self.k):
					centroid = self.centroids[centroidIndex]
					dist = self.calcEuclideanDistance(centroid, heatmiser)

					# Update smaller distance
					if dist < minDist:
						closestCluster = centroidIndex
						minDist = dist

				# Update centroid average
				self.classification[key] = closestCluster
				self.updateClusterCentroid(closestCluster)

			# See if point is closer to another cluster
			else:
				currCluster = self.classification[key]
				newCluster = None
				closestDist = self.calcEuclideanDistance(heatmiser, self.centroids[currCluster])

				for centroidIndex in range(self.k):
					# Skip if same cluster
					if centroidIndex != currCluster:
						centroid = self.centroids[centroidIndex]
						dist = self.calcEuclideanDistance(centroid, heatmiser)

						# Smaller distance found
						if dist < closestDist:
							newCluster = centroidIndex
							closestDist = dist

				# Update classification if smaller distance found
				if newCluster is not None:
					self.classification[key] = newCluster
					self.hasUpdated = True  # indicate update occurred
					self.updateClusterCentroid(currCluster)  # remove point
					self.updateClusterCentroid(newCluster)  # add point

	# Get error sum of squares of a given centroid, sum of square differences between
	# each observation and its group's mean
	def getSSE(self, index):
		currSum = 0
		centroid = self.centroids[index]
		for key in self.classification:
			# Get error of points only in given cluster
			if self.classification[key] == index:
				point = self.keyToPoint(key)
				# Difference is squared distance
				diff = ((centroid[0] - point[0]) ** 2) + ((centroid[1] - point[1]) ** 2)
				currSum += diff

		return currSum

	def getFinalStatsKMeans(self):
		lstSSE = []
		totals = {}
		print("\n")

		# Create text file to write to. Overwrites same named file
		f = open(self.fileName, "a")
		f.write("Final cluster data: \n")

		for i in range(self.k):
			SSE = self.getSSE(i)
			lstSSE.append(SSE)

			# Get total of how many osha and location instances are in a cluster
			statusOSHA = {}
			statusLocation = {}
			points = [key for key in self.classification if self.classification[key] == i]
			for point in points:
				osha = self.oshaStatus[point]
				if osha not in statusOSHA:
					statusOSHA[osha] = 1
				else:
					statusOSHA[osha] += 1

				location = self.locationStatus[point]
				if location not in statusLocation:
					statusLocation[location] = 1
				else:
					statusLocation[location] += 1

				# Increment classification totals
				if osha not in totals:
					totals[osha] = 1
				else:
					totals[osha] += 1
				if location not in totals:
					totals[location] = 1
				else:
					totals[location] += 1

			strStatusOSHA = ", ".join([(s + ": " + str(statusOSHA[s])) for s in statusOSHA])  # get string list format
			strStatusLocation = ", ".join([(s + ": " + str(statusLocation[s])) for s in statusLocation])
			res = (
				"*** --- *** \n"
				+ ("Cluster " + (str(i + 1))) + "\n"
				+ ("Center: " + ", ".join([str(p) for p in self.centroids[i]])) + "\n"
				+ ("OSHA statuses: " + strStatusOSHA) + "\n"
				+ ("Location statuses: " + strStatusLocation) + "\n"
				+ ("SSE: " + str(SSE)) + "\n"
				+ "*** --- *** \n"
			)

			f.write(res)
			print(res)

		strTotals = ", ".join([(s + ": " + str(totals[s])) for s in totals])
		f.write(("OSHA and Location status totals: " + strTotals))
		print("OSHA and Location status totals: ", strTotals)
		f.close()

		return lstSSE

	# Displays points and their respective status
	# Green = Compliant, red = NonCompliant, Cyan = Safe
	def getPointVisualizationOSHA(self, data):
		print("Processing data...")
		points = self.processData(data)

		colors = ["g", "r", "c"]
		statuses = ["Compliant", "NonCompliant", "Safe"]

		print("Visualizing data...")
		for heatmiser in points:
			status = self.oshaStatus[self.pointToKey(heatmiser)]
			color = colors[statuses.index(status)]
			plt.scatter(heatmiser[0], heatmiser[1], marker="o", color=color, s=50, linewidths=5)

		print("Visulization displayed.")
		plt.ylabel("Speed")
		plt.xlabel("Distance")
		plt.title("OSHA Statuses of Heatmiser Data")
		plt.show()

	# Displays points and their respective location
	# Yellow = Office, blue = Warehouse
	def getPointVisualizationLocation(self, data):
		print("Processing data...")
		points = self.processData(data)

		colors = ["y", "b"]
		statuses = ["Office", "Warehouse"]

		print("Visualizing data...")
		for heatmiser in points:
			status = self.locationStatus[self.pointToKey(heatmiser)]
			color = colors[statuses.index(status)]
			plt.scatter(heatmiser[0], heatmiser[1], marker="o", color=color, s=50, linewidths=5)

		print("Visulization displayed.")
		plt.ylabel("Speed")
		plt.xlabel("Distance")
		plt.title("Location Statuses of Heatmiser Data")
		plt.show()

	# Displays cluster visualization
	def getClusterVisualization(self):
		colors = self.k * ["g", "r", "c", "b", "y", "m"]

		# Display centroids and corresponding points
		for index in self.centroids:
			color = colors[index]
			centroid = self.centroids[index]
			# plt.scatter(centroid[0], centroid[1], s=150, marker="x", color="k", linewidths=5)

			# Display points
			for heatmiser in self.classification:
				if self.classification[heatmiser] == index:
					point = self.keyToPoint(heatmiser)
					plt.scatter(point[0], point[1], marker="o", color=color, s=50, linewidths=5)
			plt.scatter(centroid[0], centroid[1], s=150, marker="x", color="k", linewidths=5)

		print("Visualization displayed.")
		plt.ylabel("Speed")
		plt.xlabel("Distance")
		plt.title("K Means Clustering of Heatmiser Data")
		plt.show()

	def getDistorian(self, data):
		print("Processing data...")
		X = self.processData(data)
		print("Initializing clusters...")
		self.initClustersRandom(X)
		print("Fitting points...")

		# Restrict iterations based on hard-coded data
		first = True  # indicates initialization
		i = 0
		while self.hasUpdated and (i < self.maxIter):
			i += 1
			# Print to one line dynamically
			print("On iteration " + str(i) + " out of " + str(self.maxIter) + "                       ")
			self.fit(X, first)
			first = False

		distortion = sum(self.getFinalStatsKMeans())
		return distortion

	def run(self, data):
		print("Processing data...")
		X = self.processData(data)
		print("Initializing clusters...")

		# Initialize clusters based on distance if 2, else randomly
		if self.k == 2:
			self.initClustersFurthest(X)
		else:
			self.initClustersRandom(X)

		print("Initial clusters: ")
		print(self.centroids)
		print("Fitting points...")

		# Restrict iterations based on hard-coded data
		first = True  # indicates initialization
		i = 0
		while self.hasUpdated and (i < self.maxIter):
			i += 1
			# Print to one line dynamically
			print("On iteration " + str(i) + " out of " + str(self.maxIter))
			self.fit(X, first)
			first = False
			print("")
			print("Updated clusters: ")
			print(self.centroids)

		print("Final stats: ")
		self.getFinalStatsKMeans()
		print("Visualizing clusters...")
		self.getClusterVisualization()


class Node:
	def __init__(self):
		self.type = None  # types: leaf, feature, label
		self.label = ""
		self.parentDecision = None
		self.oldFeatures = []
		self.parent = None
		self.children = []
		self.depth = 0

	def setType(self, type):
		self.type = type

	def getType(self):
		return self.type

	def setLabel(self, label):
		self.label = label

	def getLabel(self):
		return self.label

	def setParentDecision(self, decision):
		self.parentDecision = decision

	def getParentDecision(self):
		return self.parentDecision

	def setOldFeatures(self, features):
		self.oldFeatures = features

	def appendOldFeatures(self, feature):
		self.oldFeatures.append(feature)

	def getOldFeatures(self):
		return self.oldFeatures

	def setParent(self, parent):
		self.parent = parent

	def getParent(self):
		return self.parent

	def setChildren(self, children):
		self.children = children

	def appendChild(self, child):
		self.children.append(child)

	def getChildren(self):
		return self.children

	def setDepth(self, depth):
		self.depth = depth

	def incrementDepth(self):
		self.depth += 1

	def getDepth(self):
		return self.depth


# Optimized HeatMiser class that adjusts humidity and temp to comfortable levels
class DecisionTree:
	def __init__(self):
		self.rootNode = None

	def setRoot(self, root):
		self.rootNode = root

	def getRoot(self):
		return self.rootNode

	'''
		Helper function that gets counts of values for each feature
	'''

	def getFeatureTotals(self, data, node):
		totalsDict = {}

		# Iterate through each heatmiser
		for i in range(0, len(data['ID'])):
			# Iterate through each feature in heatmiser data
			complianceStatus = data['OSHA'][i]  # get compliance status of heatmiser
			for feature in data:
				# Alternative OSHA dictionary
				if (feature == 'OSHA'):
					if feature not in totalsDict:
						totalsDict[feature] = {'total': 0}

					value = data[feature][i]
					if value not in totalsDict[feature]:
						totalsDict[feature][value] = 0

					totalsDict[feature][value] += 1
					totalsDict[feature]['total'] += 1

				elif (feature != 'ID' and feature not in node.getOldFeatures()):
					# Add variable if not in dictionary
					if feature not in totalsDict:
						totalsDict[feature] = {'total': 0}

					# Increment feature's value count
					value = data[feature][i]
					# Initialize classification if not present
					if value not in totalsDict[feature]:
						totalsDict[feature][value] = {}
						totalsDict[feature][value]['total'] = 0
						totalsDict[feature][value]['osha_status'] = {}
						totalsDict[feature][value]['osha_status']['Safe'] = 0
						totalsDict[feature][value]['osha_status']['Compliant'] = 0
						totalsDict[feature][value]['osha_status']['NonCompliant'] = 0

					totalsDict[feature][value]['total'] += 1
					totalsDict[feature]['total'] += 1
					totalsDict[feature][value]['osha_status'][complianceStatus] += 1

		# print(totalsDict)
		# print("")
		return totalsDict

	def print_tree(self, node, string):
		nodes = node.getChildren().copy()
		if (node.getType() == 'leaf'):
			string += "\n Compliance Result: -> | " + node.getLabel() + " | <-\n"
		else:
			string += node.getType() + ": " + node.getLabel() + "--> "

		while nodes:
			child = nodes.pop()
			self.print_tree(child, string)

		if (node.getType() == 'leaf'):
			print(string)

	def start_tree(self, data):
		root = Node()
		self.setRoot(root)
		root.setType('feature')

		highestGain = self.setInformationGain(data, root)

		root.setLabel(highestGain)

		root.incrementDepth()
		self.buildTree(data, root)

	def buildTree(self, data, node):
		labels = {}
		labels["Speeding"] = ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast']
		labels["Distance"] = ['Very Close', 'Close', 'Neutral', 'Far', 'Very Far']
		labels["Location"] = ['Office', 'Warehouse']

		if (node.getType() == 'feature'):
			for label in labels[node.getLabel()]:
				child = Node()
				node.appendChild(child)

				child.setParent(node)
				child.setType('label')
				child.setLabel(label)
				child.setDepth(node.getDepth())
				child.incrementDepth()
				child.setOldFeatures(node.getOldFeatures().copy())
				child.appendOldFeatures(node.getLabel())
				child.setParentDecision(node.getParentDecision())

				isolatedData = self.isolate_label(data, node.getLabel(), label)

				self.buildTree(isolatedData, child)

		elif (node.getType() == 'label'):
			highestGain = self.setInformationGain(data, node)

			if highestGain == None:
				return node

			child = Node()
			node.appendChild(child)

			child.setParent(node)
			type(child.getParent())
			child.setLabel(highestGain)
			child.setDepth(node.getDepth())
			child.incrementDepth()
			child.setOldFeatures(node.getOldFeatures().copy())
			child.setParentDecision(node.getParentDecision())

			if highestGain in ["Safe", "Compliant", "NonCompliant"]:
				child.setType('leaf')
				return child

			else:
				child.setType('feature')
				self.buildTree(data, child)

		return node

	'''
	Returns data set containing only data with specific key label pair
	'''

	def isolate_label(self, data, key, label):
		labelData = {}
		ids = []
		distances = []
		speeds = []
		locations = []
		oshas = []
		headers = ['ID', 'Distance', 'Speeding', 'Location', 'OSHA']

		for i in range(0, len(data['ID'])):
			if (data[key][i] == label):
				ids.append(data['ID'][i])
				distances.append(data['Distance'][i])
				speeds.append(data['Speeding'][i])
				locations.append(data['Location'][i])
				oshas.append(data['OSHA'][i])

		labelData[headers[0]] = ids
		labelData[headers[1]] = distances
		labelData[headers[2]] = speeds
		labelData[headers[3]] = locations
		labelData[headers[4]] = oshas

		return labelData

	'''
	Helper function to calculate entropy
	'''

	def getFeatureEntropy(self, valueDict):
		entropy = 0
		# Get log sum
		for compliance in valueDict['osha_status']:
			prob = valueDict['osha_status'][compliance] / valueDict['total']

			# Skip if 0 encountered
			if prob != 0.0:
				log = -(prob * (math.log(prob, 2.0)))
				entropy += log

		return entropy

	'''
	Helper function to calculate class entropy
	'''

	def getClassEntropy(self, classDict):
		entropy = 0
		# Get log sum
		for status in classDict:
			if (status == 'total'):
				continue

			prob = classDict[status] / classDict['total']
			log = -(prob * (math.log(prob, 2.0)))
			entropy += log

		return entropy

	'''
	Calculates information entropy of a feature
	'''

	def getFeatureInfoEntropy(self, featureDict):
		infoEntropy = 0
		# Iterate through each value of a feature
		for value in featureDict:
			if value == 'total':
				continue

			prob = featureDict[value]['total'] / featureDict['total']  # get probability
			entropy = self.getFeatureEntropy(featureDict[value])  # get entropy of value
			infoEntropy += (prob * entropy)

		return infoEntropy

	def get_max(self, featureTotals):
		featureTotalsCopy = featureTotals['OSHA'].copy()
		featureTotalsCopy.pop('total')
		return max(featureTotalsCopy, key=lambda key: featureTotalsCopy[key])

	''' 
	If the tree doesn't end in a perfect leaf node, then create one by majority of OSHA label
	'''

	def setMajority(self, featureTotals, node):
		maxGain = self.get_max(featureTotals)

		child = Node()
		child.setType('leaf')
		node.appendChild(child)

		child.setLabel(maxGain)
		child.setParent(node)
		return None

	'''
	Calculates information gain of all features
	'''

	def setInformationGain(self, data, node):
		# print(node.getLabel())
		allInfoGain = {}
		featureTotals = self.getFeatureTotals(data, node)

		if (len(featureTotals.keys()) == 1):
			return self.setMajority(featureTotals, node)

		if (featureTotals):
			classEntropy = self.getClassEntropy(featureTotals['OSHA'])

			if (classEntropy <= 0.0):
				child = Node()
				node.appendChild(child)

				child.setType('leaf')
				for key in featureTotals['OSHA'].keys():
					if key != 'total':
						child.setLabel(key)

				child.setParent(node)
				return None

			elif (node.getType() == 'label'):
				node.setParentDecision(self.get_max(featureTotals))
		else:
			return node.getParentDecision()

		for feature in featureTotals:
			# pass if OSHA
			if (feature == 'OSHA'):
				continue

			featureEntropy = self.getFeatureInfoEntropy(featureTotals[feature])
			informationGain = classEntropy - featureEntropy
			allInfoGain[feature] = informationGain

		maxGain = max(allInfoGain, key=lambda key: allInfoGain[key])

		if (len(allInfoGain) == 1 and allInfoGain[maxGain] == 0.0):
			return self.setMajority(featureTotals, node)

		return maxGain

	def predict(self, test_data, test_number, results):
		curr = self.getRoot()

		while (curr.getType() != 'leaf'):
			if (curr.getType() == 'feature'):
				if (len(curr.getChildren()) == 1):
					curr = curr.getChildren()[0]

				else:
					for child in curr.getChildren():
						if (child.getLabel() == test_data[curr.getLabel()][test_number]):
							curr = child
							break

			else:
				curr = curr.getChildren()[0]

		# Label doesn't match, increment false positive
		if (curr.getLabel() != test_data['OSHA'][test_number]):
			# print("Predicted OSHA status of HeatMiser " + test_data['ID'][test_number] + " is: "
			#       + curr.getLabel())
			#
			# print("Actual OSHA status is: " + test_data['OSHA'][test_number] + "\n")

			# Update false positive: labels
			if (test_data['OSHA'][test_number] == 'Compliant'):
				results["Errors"]["Compliant"] += 1

			elif (test_data['OSHA'][test_number] == 'NonCompliant'):
				results["Errors"]["NonCompliant"] += 1

			elif (test_data['OSHA'][test_number] == 'Safe'):
				results["Errors"]["Safe"] += 1

			# Update false negative: incorrect label 
			results["FalseNegatives"][(test_data['OSHA'][test_number])] += 1

			results["Errors"]["Total"] += 1
			results["FalseNegatives"]["Total"] += 1

		# Update totals
		if (test_data['OSHA'][test_number] == 'Compliant'):
			results["Totals"]["Compliant"] += 1

		elif (test_data['OSHA'][test_number] == 'NonCompliant'):
			results["Totals"]["NonCompliant"] += 1

		elif (test_data['OSHA'][test_number] == 'Safe'):
			results["Totals"]["Safe"] += 1

		return results

	'''
	Make predictions for the OSHA status of each HeatMiser in the test data, based on decision tree
	'''
	def make_all_predictions(self, test_data):
		results = {}
		results["Totals"] = {}
		results["Errors"] = {}
		results["FalseNegatives"] = {}

		results["Totals"]["Compliant"] = 0
		results["Totals"]["Safe"] = 0
		results["Totals"]["NonCompliant"] = 0

		# False Positives
		results["Errors"]["Compliant"] = 0
		results["Errors"]["Safe"] = 0
		results["Errors"]["NonCompliant"] = 0
		results["Errors"]["Total"] = 0

		# False Negatives
		results["FalseNegatives"]["Compliant"] = 0
		results["FalseNegatives"]["Safe"] = 0
		results["FalseNegatives"]["NonCompliant"] = 0
		results["FalseNegatives"]["Total"] = 0

		for i in range(400):
			results = self.predict(test_data, i, results)

		# print(str(results["Errors"]["Total"]) + " wrong out of 4000 tests, or accuracy of "
		#       + str(((400 - results["Errors"]["Total"]) / 400) * 100) + "%.\n")

		return results


'''
Helper function to split into folds for cross validation
'''
def split_into_folds(data):
	foldLabels = ["Fold One", "Fold Two", "Fold Three", "Fold Four", "Fold Five",
				  "Fold Six", "Fold Seven", "Fold Eight", "Fold Nine", "Fold Ten"]
	folds = {}
	counter = 0
	start = 0
	end = 400

	while (counter < 10):
		newData = {}
		ids = []
		ids = data['ID'][start:end]
		newData['ID'] = ids

		distances = []
		distances = data['Distance'][start:end]
		newData['Distance'] = distances

		speeds = []
		speeds = data['Speeding'][start:end]
		newData['Speeding'] = speeds

		locations = []
		locations = data['Location'][start:end]
		newData['Location'] = locations

		oshas = []
		oshas = data['OSHA'][start:end]
		newData['OSHA'] = oshas

		folds[foldLabels[counter]] = newData

		counter += 1
		start += 400
		end += 400

	# print(folds)
	return folds


def isolate_training_data(data, testing_fold):
	training_data = {}
	ids = []
	distances = []
	speeds = []
	locations = []
	oshas = []

	start = 0
	end = 400
	first = True

	for i in range(10):
		if (i != testing_fold):
			if (first):
				distances = data['Distance'][start:end]
				speeds = data['Speeding'][start:end]
				first = False
			else:
				ids = ids + data['ID'][start:end]
				distances = distances + data['Distance'][start:end]
			speeds = speeds + data['Speeding'][start:end]
			locations = locations + data['Location'][start:end]
			oshas = oshas + data['OSHA'][start:end]

		start += 400
		end += 400

	training_data['ID'] = ids
	training_data['Distance'] = distances
	training_data['Speeding'] = speeds
	training_data['Location'] = locations
	training_data['OSHA'] = oshas

	# print(training_data)
	return training_data


def isolate_testing_data(data, test_start, test_end):
	testing_data = {}

	testing_data['ID'] = data['ID'][test_start:test_end]
	testing_data['Distance'] = data['Distance'][test_start:test_end]
	testing_data['Speeding'] = data['Speeding'][test_start:test_end]
	testing_data['Location'] = data['Location'][test_start:test_end]
	testing_data['OSHA'] = data['OSHA'][test_start:test_end]

	# print(testing_data)
	return testing_data


def update_results(results, newResult):
	results["Totals"]["Compliant"] += newResult["Totals"]["Compliant"]
	results["Totals"]["NonCompliant"] += newResult["Totals"]["NonCompliant"]
	results["Totals"]["Safe"] += newResult["Totals"]["Safe"]

	results["Errors"]["Compliant"] += newResult["Errors"]["Compliant"]
	results["Errors"]["NonCompliant"] += newResult["Errors"]["NonCompliant"]
	results["Errors"]["Safe"] += newResult["Errors"]["Safe"]
	results["Errors"]["Total"] += newResult["Errors"]["Total"]

	results["FalseNegatives"]["Compliant"] += newResult["FalseNegatives"]["Compliant"]
	results["FalseNegatives"]["NonCompliant"] += newResult["FalseNegatives"]["NonCompliant"]
	results["FalseNegatives"]["Safe"] += newResult["FalseNegatives"]["Safe"]

	return results


'''
Runs Decision Tree training and testing 10 times to ensure validity 
'''
def ten_fold_cross_validation(data):
	# folds = dt.split_into_folds(data)
	test_fold = 0
	results = {}

	results["Totals"] = {}
	results["Errors"] = {}
	results["FalseNegatives"] = {}

	results["Totals"]["Compliant"] = 0
	results["Totals"]["Safe"] = 0
	results["Totals"]["NonCompliant"] = 0

	# False Positives
	results["Errors"]["Compliant"] = 0
	results["Errors"]["Safe"] = 0
	results["Errors"]["NonCompliant"] = 0
	results["Errors"]["Total"] = 0

	# False Negatives
	results["FalseNegatives"]["Compliant"] = 0
	results["FalseNegatives"]["Safe"] = 0
	results["FalseNegatives"]["NonCompliant"] = 0
	results["FalseNegatives"]["Total"] = 0

	for i in range(10):
		# print("\n\n --------- START OF FOLD " + str(i) + " ---------\n")
		dt = DecisionTree()

		training_data = isolate_training_data(data, test_fold)
		dt.start_tree(training_data)

		# dt.print_tree(dt.getRoot(), "")

		test_start = (0 + (400 * i))
		test_end = (400 + (400 * i))
		test_data = isolate_testing_data(data, test_start, test_end)

		newResult = dt.make_all_predictions(test_data)

		results = update_results(results, newResult)

		test_fold += 1
	# print("\n\n --------- END OF FOLD " + str(i) + " ---------\n")

	print("Decision Tree 10 Fold Cross Validation Results")

	print("Compliant: Seen -> " + str(results["Totals"]["Compliant"]))	
	compliantTP = results["Totals"]["Compliant"] - results["Errors"]["Compliant"]
	compliantFP = results["Errors"]["Compliant"]
	compliantFN = results["FalseNegatives"]["Compliant"]
	print("		Correctly Predicted -> " + str(compliantTP))
	print("		False Positives -> " + str(compliantFP))
	print("		False Negatives -> " + str(compliantFN) + "\n")
	print("		Accuracy -> " + str((compliantTP / results["Totals"]["Compliant"]) * 100) + "%")
	# Catch division by 0
	if ((compliantTP == 0) and (compliantFP == 0)):
		compliantPrecision = 0
	else:
		compliantPrecision = (compliantTP / (compliantTP + compliantFP))*100
	print("		Precision -> " + str(compliantPrecision) + "%")
	
	if ((compliantTP == 0) and (compliantFN == 0)):
		compliantRecall = 0
	else:
		compliantRecall = (compliantTP / (compliantTP + compliantFN))*100
	print("		Recall -> " + str(compliantRecall) + "%")
	if ((compliantRecall == 0) and (compliantPrecision == 0)):
		compliantF1 = 0
	else:
		compliantF1 = 2*((compliantPrecision * compliantRecall) / (compliantPrecision + compliantRecall))
	print("		F1-measure -> " + str(compliantF1) + "\n")


	print("NonCompliant: Seen -> " + str(results["Totals"]["NonCompliant"]))
	nonCompliantTP = results["Totals"]["NonCompliant"] - results["Errors"]["NonCompliant"]
	nonCompliantFP = results["Errors"]["NonCompliant"]
	nonCompliantFN = results["FalseNegatives"]["NonCompliant"]
	print("		Correctly Predicted -> " + str(nonCompliantTP))
	print("		False Positives -> " + str(nonCompliantFP))
	print("		False Negatives -> " + str(nonCompliantFN) + "\n")
	print("		Accuracy -> " + str((nonCompliantTP / results["Totals"]["NonCompliant"]) * 100) + "%")
	# Catch division by 0
	if ((nonCompliantTP == 0) and (nonCompliantFP == 0)):
		nonCompliantPrecision = 0
	else:
		nonCompliantPrecision = (nonCompliantTP / (nonCompliantTP + nonCompliantFP))*100
	print("		Precision -> " + str(compliantPrecision) + "%")
	
	if ((nonCompliantTP == 0) and (nonCompliantFN == 0)):
		nonCompliantRecall = 0
	else:
		nonCompliantRecall = (nonCompliantTP / (nonCompliantTP + nonCompliantFN))*100
	print("		Recall -> " + str(compliantRecall) + "%")
	if ((nonCompliantRecall == 0) and (nonCompliantPrecision == 0)):
		nonCompliantF1 = 0
	else:
		nonCompliantF1 = 2*((nonCompliantPrecision * nonCompliantRecall) / (nonCompliantPrecision + nonCompliantRecall))
	print("		F1-measure -> " + str(nonCompliantF1) + "\n")


	print("Safe: Seen -> " + str(results["Totals"]["Safe"]))
	safeTP = results["Totals"]["Safe"] - results["Errors"]["Safe"]
	safeFP = results["Errors"]["Safe"]
	safeFN = results["FalseNegatives"]["Safe"]
	print("		Correctly Predicted -> " + str(safeTP))
	print("		False Positives -> " + str(safeFP))
	print("		False Negatives -> " + str(safeFN) + "\n")
	print("		Accuracy -> " + str((safeTP / results["Totals"]["Safe"]) * 100) + "%")
	# Catch division by 0
	if ((safeTP == 0) and (safeFP == 0)):
		safePrecision = 0
	else:
		safePrecision = (safeTP / (safeTP + safeFP))*100
	print("		Precision -> " + str(safePrecision) + "%")
	
	if ((safeTP == 0) and (safeFN == 0)):
		safeRecall = 0
	else:
		safeRecall = (safeTP / (safeTP + safeFN))*100
	print("		Recall -> " + str(safeRecall) + "%")
	if ((safeRecall == 0) and (safePrecision == 0)):
		safeF1 = 0
	else:
		safeF1 = 2*((safePrecision * safeRecall) / (safePrecision + safeRecall))
	print("		F1-measure -> " + str(safeF1) + "\n")


	print("Overall Accuracy -> " + str((4000 - results["Errors"]["Total"]) / 4000 * 100) + "%")



# Function to determine optimal K
def determineOptimalK(data):
	# Create a new plot
	plt.plot()
	distortions = []
	# Display graph up to k = 10
	K = range(1, 11)
	for k in K:
		kModel = KMeansClustering(k)
		distortion = kModel.getDistorian(data)
		distortions.append(distortion)

	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum of squared errors')
	plt.title('The Elbow Method showing the optimal k')
	plt.show()


# Shows plot of all heatmiser in data colored by their respective OSHA status
# 1 shows OSHA, 2 shows location type
def getStatusPlot(data, feature):
	plt.plot()  # create a new plot
	kModel = KMeansClustering(1)
	if feature == 1:
		kModel.getPointVisualizationOSHA(data)
	elif feature == 2:
		kModel.getPointVisualizationLocation(data)


def main():
	dt = DecisionTree()
	data = dp.getDataArrays()
	ten_fold_cross_validation(data)

	# searchType = input("Welcome to OSHA Compliant HeatMiser! \nPlease select your option by pressing the appropriate number:"
	# 	+ " 1 for decision tree, 2 for k means clustering, or 3 to quit: ")
	# if (searchType == "1"):
	# 	print("Decision tree approach selected!")
	# 	search = 0
	# elif (searchType == "2"):
	# 	print("K means clustering approach selected!")
	# data = dp.getDataList()
	# kCluster = KMeansClustering(2)
	# kCluster.run(data)
	# elif (searchType == "3"):
	# 	print("Shutting down...")
	# else:
	# 	print("Sorry, that was an incorrect command. Shutting down...")
	# 	sys.exit()

	# Uncomment below to get either OSHA (1) or location (2) plot
	# getStatusPlot(data, 2)

	# Uncomment below to run elbow method
	# determineOptimalK(data)

if __name__ == '__main__':
	main()
