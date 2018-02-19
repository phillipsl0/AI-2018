'''
OSHACompliantHeatMiser.py
Assignment #3
'''

import DataParser as dp
import sys
import math

# Optimized HeatMiser class that adjusts humidity and temp to comfortable levels
class DecisionTree:
    def __init__(self):
        self.allInfoGain = {}

    # def getFeatureTotals(self, data):
    #     totalsDict = {}
    #     # Iterate through each heatmiser
    #     for heatmiser in data:
    #         # Iterate through each feature in heatmiser data
    #         complianceStatus = data['OSHA'] # get compliance status of heatmiser
    #         for feature in heatmiser:
    #             # Alternative OSHA dictionary
    #             if (feature == 'OSHA'):
    #                 if feature not in totalsDict:
    #                     totalsDict[feature] = {'total': 0}
    #
    #                 value = heatmiser[feature]
    #                 if value not in totalsDict[feature]:
    #                     totalsDict[feature][value] = 0
    #
    #                 totalsDict[feature][value] += 1
    #                 totalsDict[feature]['total'] += 1
    #
    #             else:
    #                 # Add variable if not in dictionary
    #                 if feature not in totalsDict:
    #                     totalsDict[feature] = {'total': 0}
    #
    #                 # Increment feature's value count
    #                 value = heatmiser[feature]
    #                 # Initialize classification if not present
    #                 if value not in totalsDict[feature]:
    #                     totalsDict[feature][value] = {}
    #                     totalsDict[feature][value]['total'] = 0
    #                     totalsDict[feature][value]['osha_status'] = {}
    #                     totalsDict[feature][value]['osha_status']['Safe'] = 0
    #                     totalsDict[feature][value]['osha_status']['Compliant'] = 0
    #                     totalsDict[feature][value]['osha_status']['NonCompliant'] = 0
    #
    #                 totalsDict[feature][value]['total'] += 1
    #                 totalsDict[feature]['total'] += 1
    #                 totalsDict[feature][value]['osha_status'][complianceStatus] += 1
    #
    #     print(totalsDict)
    #     return totalsDict

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


    # '''
    # Creates splits, an attribute in the dataset and a value, useful for indexing into rows of data
    # Separating dataset into two rows given index of an attribute and a split value for the attribute
    # Split a dataset based on an attribute and an attribuet value
    # '''
    # def splitData(self, index, value, dataset):
    # 	left = []
    # 	right = []

    # 	# Iterate over each row in dataset
    # 	for row in dataset:
    # 		# If attribute below split value, assign to left
    # 		if row[index] < value:
    # 			left.append(row)
    # 		# If above or equal to, assign to right
    # 		else:
    # 			right.append(row)

    # 	return left, right


    # '''
    # Select best split point for data
    # '''
    # def getSplit(self, dataset):
    # 	classValues = list(set(row[-1] for row in dataset))
    # 	bIndex, bValue, bScore = 999, 999, 999
    # 	bGroups = None

    # 	# Iterate over each attribute as a candidate split
    # 	for index in range(len(dataset[0])-1):
    # 		print("On index: " + str(index))
    # 		# Check every value of that attribute
    # 		count = 0
    # 		for row in dataset:
    # 			count += 1
    # 			sys.stdout.write("On row " + str(count) + " out of " + str(len(dataset)) + "                         \r",)
    # 			sys.stdout.flush()


    # 			groups = self.splitData(index, row[index], dataset)
    # 			gini = self.calculateGiniIndex(groups, classValues)

    # 			# update best split point if new node is better
    # 			if gini < bScore:
    # 				#print("Better gini index: " + str(gini))
    # 				#print(groups)
    # 				bIndex = index
    # 				bValue = row[index]
    # 				bSore = gini
    # 				bGroups = groups

    # 	# Use dictionary to represent node in the decision tree
    # 	return {'index': bIndex, 'value': bValue, 'groups': bGroups}

    # '''
    # Cost function used to evaluate splits in the dataset
    # '''
    # def calculateGiniIndex(self, groups, classes):
    # 	# count all samples at split point
    # 	n_instances = float(sum([len(group) for group in groups]))
    # 	# sum weighted Gini index for each group
    # 	gini = 0.0
    # 	for group in groups:
    # 		size = float(len(group))
    # 		# avoid divide by zero
    # 		if size == 0:
    # 			continue
    # 		score = 0.0
    # 		# score group based on the score for each class
    # 		for class_val in classes:
    # 			p = [row[-1] for row in group].count(class_val) / size
    # 			score += p * p
    # 		# weight the group score by its relative size
    # 		gini += (1.0 - score) * (size / n_instances)
    # 	return gini


    # # Print precision, recall, and F1 of each individual class and all classes combined
    # # Output for each fold and average of all 10 folds
    # # Output plot of performance for each fold over majority class baseline
    # def printResults(self):
    # 	pass


def main():
    dt = DecisionTree()
    data = dp.getDataArrays()
    info = dt.setInformationGain(data)
    # data = dp.getDataList()
    # split = dt.getSplit(data)
    # print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))

if __name__ == '__main__':
    main()
