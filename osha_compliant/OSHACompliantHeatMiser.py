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
		self.k = k # number of clusters
		self.centroids = {}
		self.classification = {} # dictionary containing key of a dist, speed point and value of cluster index it belongs to
		self.maxIter = maxIter
		self.hasUpdated = True
		self.oshaStatus = {}
		self.fileName = fileName

	# Converts heatmiser arrays into a numpy array of distance, speed values
	# Gets heatmiser status.
	def processData(self, data):
		arr = []
		for heatmiser in data:
			point = [float(heatmiser[1]), float(heatmiser[2])]
			arr.append(point)
			self.oshaStatus[self.pointToKey(point)] = heatmiser[4] # Get OSHA status for heatmiser
		return arr

	def calcEuclideanDistance(self, p, q):
		pX = p[0]
		pY = p[1]

		qX = q[0]
		qY = q[1]

		return math.sqrt(((qX - pX)**2) + ((qY - pY)**2))

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
			sys.stdout.write("Fitting point " + str(count) + " out of " + str(len(points)) + "                                                \r",)
			sys.stdout.flush()

			key = self.pointToKey(heatmiser)
			# If initializing, assign point to nearest cluster
			if first:
				self.hasUpdated = True # indicates update will happen
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
					self.hasUpdated = True # indicate update occurred
					self.updateClusterCentroid(currCluster) # remove point
					self.updateClusterCentroid(newCluster) # add point


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
				diff = ((centroid[0] - point[0])**2) + ((centroid[1] - point[1])**2)
				currSum += diff

		return currSum

	def getFinalStatsKMeans(self):
		lstSSE = []
		totals = {}
		print("\n")

		f = open(self.fileName, "a")
		f.write("Final cluster data: \n")

		for i in range(self.k):
			SSE = self.getSSE(i)
			lstSSE.append(SSE)

			# Get total of how many osha instances are in a cluster
			status = {}
			points = [key for key in self.classification if self.classification[key] == i]
			for point in points:
				osha = self.oshaStatus[point]
				if osha not in status:
					status[osha] = 1
				else:
					status[osha] += 1

				# Increment classification totals
				if osha not in totals:
					totals[osha] = 1
				else:
					totals[osha] += 1

			strStatus = ", ".join([(s + ": " + str(status[s])) for s in status]) # get string list format
			res = (
				"*** --- *** \n" 
			+ ("Cluster " + (str(i))) + "\n"
			+ ("Center: " + ", ".join([str(p) for p in self.centroids[i]])) + "\n"
			+ ("OSHA statuses: " + strStatus) + "\n"
			+ ("SSE: " + str(SSE)) + "\n"
			+  "\n*** --- *** \n"
			)

			f.write(res)
			print(res)

		strTotals = ", ".join([(s + ": " + str(totals[s])) for s in totals])
		f.write(("OSHA status totals: " + strTotals))
		print("OSHA status totals: ", strTotals)
		f.close()

		return lstSSE

	# Displays cluster visualization
	def getClusterVisualization(self, points):
		colors = ["g","r","c","b","y"]
		
		# Display centroids and corresponding points
		for index in self.centroids:
			color = colors[index]
			centroid = self.centroids[index]
			#plt.scatter(centroid[0], centroid[1], s=150, marker="x", color="k", linewidths=5)

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
		first = True # indicates initialization
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
		# self.initClustersRandom(X) 		# initialize clusters randomly
		self.initClustersFurthest(X) 	# initialize clusters based on distance

		print("Initial clusters: ")
		print(self.centroids)
		print("Fitting points...")
		
		# Restrict iterations based on hard-coded data
		first = True # indicates initialization
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
		self.getClusterVisualization(X)



# Optimized HeatMiser class that adjusts humidity and temp to comfortable levels
class DecisionTree:
    def __init__(self):
        self.allInfoGain = {}

    '''
        Helper function that gets counts of values for each feature
    '''
    def getFeatureTotals(self, data):
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

                elif (feature != 'ID'):
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

        print(totalsDict)
        return totalsDict

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
        # print(featureDict)
        infoEntropy = 0
        # Iterate through each value of a feature
        for value in featureDict:
            if value == 'total':
                continue

            prob = featureDict[value]['total'] / featureDict['total'] # get probability
            entropy = self.getFeatureEntropy(featureDict[value]) # get entropy of value
            infoEntropy += (prob * entropy)

        return infoEntropy

    '''
    Calculates information gain of all features
    '''
    def setInformationGain(self, data):
        featureTotals = self.getFeatureTotals(data)

        classEntropy = self.getClassEntropy(featureTotals['OSHA'])

        for feature in featureTotals:
            # pass if OSHA
            if (feature == 'OSHA'):
                continue

            featureEntropy = self.getFeatureInfoEntropy(featureTotals[feature])
            informationGain = classEntropy - featureEntropy

            self.allInfoGain[feature] = informationGain

        print(self.allInfoGain)

# Function to determine optimal K
def determineOptimalK(data):
	# Create a new plot
	plt.plot()
	distortions = []
	# Display graph up to k = 10
	K = range(1, 10)
	for k in K:
		kModel = KMeansClustering(k)
		distortion = kModel.getDistorian(data)
		distortions.append(distortion)

	plt.plot(K, distortions, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum of squared errors')
	plt.title('The Elbow Method showing the optimal k')
	plt.show()


def main():
	# dt = DecisionTree()
	# data = dp.getDataJSON()
	# info = dt.setInformationGain(data)
	
	# Create text file to write to. Overwrites previous trial
	fileName = ""

	searchType = input("Welcome to OSHA Compliant HeatMiser! \nPlease select your option by pressing the appropriate number:" 
		+ " 1 for decision tree, 2 for k means clustering, or 3 to quit: ")
	if (searchType == "1"):
		print("Decision tree approach selected!")
		fileName = "osha_heatmiser_trial_output_dt"
		search = 0
	elif (searchType == "2"):
		print("K means clustering approach selected!")
		fileName = "osha_heatmiser_trial_output_kmeans"
		search = 1
	elif (searchType == "3"):
		print("Shutting down...")
	else:
		print("Sorry, that was an incorrect command. Shutting down...")
		sys.exit()
    
	f = open(fileName, "w")
	f.close()

	data = dp.getDataList()
	kCluster = KMeansClustering(2)
	kCluster.run(data)

	#determineOptimalK(data)

if __name__ == '__main__':
    main()
