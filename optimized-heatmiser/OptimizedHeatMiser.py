'''
OptimizedHeatMiser.py
Assignment #2
'''

from random import uniform, randrange
from math import sqrt
from queue import PriorityQueue
import HeuristicParser

# Room class that sets and returns its humidity and temp
class Room:
    def __init__(self, temp, humidity, index):
        self.temp = temp
        self.humidity = humidity
        self.index = index

    def setTemp(self, temp):
        self.temp = temp

    def setHumidity(self, humidity):
        self.humidity = humidity

    def getTemp(self):
        return self.temp

    def getHumidity(self):
        return self.humidity

    def getIndex(self):
        return self.index



# Floor class that generates rooms and sets average humidity and temp
class Floor:
    def __init__(self):
        self.rooms = []
        self.totalTemp = self.totalHumidity = 0
        self.avgTemp = self.avgHumidity = self.stdTemp = self.stdHumidity = 0.0
        # Array listing index of adjacent offices of each office
        self.floorPlan = {
            1: [2, 3],
            2: [1, 4],
            3: [1, 7],
            4: [2, 5, 6],
            5: [4, 8],
            6: [4, 7],
            7: [3, 6, 10],
            8: [5, 9],
            9: [8, 10],
            10: [7, 11],
            11: [10, 12],
            12: [11]
        }

        self.floorPlanCost = {
            1: {2: 13, 3: 15},
            2: {1: 13, 4: 7},
            3: {1: 15, 7: 23},
            4: {2: 7, 5: 6, 6: 10},
            5: {4: 6, 8: 4},
            6: {4: 10, 7: 9},
            7: {3: 23, 6: 9, 10: 17},
            8: {5: 4, 9: 5},
            9: {8: 5, 10: 8},
            10: {7: 17, 11: 2},
            11: {10: 2, 12: 19},
            12: {11: 19}
        }

        self.floorPlanHeuristic = HeuristicParser.getHeuristic()

        # Open created text file to append output
        f = open("heatmiser_trial_output", "a")
        f.write("Starting room states: \n")
        print("Starting room states:")
        # Generates 12 random rooms in the floor
        for i in range(12):
            # Python random for temp
            # which cannot exceed the range of 65-75 degrees or 45-55% humidity
            
            room = Room(uniform(65.0, 75.0), uniform(45.0, 55.0), i)
        
            self.totalHumidity += room.getHumidity()
            self.totalTemp += room.getTemp()
            
            # Print current state of room as they're generated
            print("Room " + str(i+1) + ": " + str("%.1f" % room.getTemp()) + "°F & " +
                  str("%.1f" %room.getHumidity()) + "%")

            # Add current state of room to text file
            f.write("Room " + str(i+1) + ": " + str("%.1f" % room.getTemp()) + "°F & " +
                  str("%.1f" %room.getHumidity()) + "%" + "\n")

            self.rooms.append(room)

        print("")
        f.close() # Close file
        self.calculateAverageTemp()
        self.calculateAverageHumidity()

    def getAverageTemp(self):
        return self.avgTemp

    def getAverageHumidity(self):
        return self.avgHumidity

    def getRooms(self):
        return self.rooms

    def getFloorPlan(self):
        return self.floorPlan

    def getFloorCost(self):
        return self.floorPlanCost

    def getHeuristicCost(self):
        return self.floorPlanHeuristic

    def getStandardDeviationTemp(self):
        return self.stdTemp

    def getStandardDeviationHumidity(self):
        return self.stdHumidity

    def setRoomTemp(self, index, temp):
        oldTemp = self.rooms[index].getTemp()
        self.rooms[index].setTemp(temp)
        self.updateTotalTemp(oldTemp, temp)
        self.calculateAverageTemp()

    def setRoomHumidity(self, index, humidity):
        oldHumidity = self.rooms[index].getHumidity()
        self.rooms[index].setHumidity(humidity)
        self.updateTotalHumidity(oldHumidity, humidity)
        self.calculateAverageHumidity()

    def updateTotalTemp(self, oldTemp, temp):
        self.totalTemp -= oldTemp
        self.totalTemp += temp

    def updateTotalHumidity(self, oldHumidity, humidity):
        self.totalHumidity -= oldHumidity
        self.totalHumidity += humidity

    def calculateAverageHumidity(self):
        self.avgHumidity = self.totalHumidity / 12

    def calculateAverageTemp(self):
        self.avgTemp = self.totalTemp / 12

    # Calculates standard deviation for both humidity and temp of floor
    def calculateStandardDeviation(self):
        xTemp = 0
        xHumidity = 0

        for room in self.rooms:
            xTemp += ((room.getTemp() - self.avgTemp)**2)
            xHumidity += ((room.getHumidity() - self.avgHumidity)**2)

        self.stdTemp = sqrt(xTemp / 12)
        self.stdHumidity = sqrt(xHumidity / 12)



# Optimized HeatMiser class that adjusts humidity and temp to comfortable levels
class OptimizedHeatMiser:
    def __init__(self, floor, trial):
        self.floor = floor
        self.raiseTemp = self.raiseHumidity = None
        self.visits = 0
        self.energyUse = 0
        self.trial = trial

    # Determine office with max difference of specified setting; 0 for temp, 1 for humidity
    def maxDiff(self, setting):
        rooms = self.floor.getRooms()

        maxRoom = None
        maxDiff = -1
        # Iterate through room list to determine max val
        for r in rooms:
            diff = None

            if setting == 0:
                diff = abs(r.getTemp() - 72)
            elif setting == 1:
                diff = abs(r.getHumidity() - 47)
            else:
                print("This shouldn't happen")

            if diff > maxDiff:
                maxDiff = diff
                maxRoom = r
        return maxRoom

    # Return index of room with max differene in temperature
    def getRoomMaxDiffTemperature(self):
        return self.maxDiff(0)

    # Return index of room with max different in humidity
    def getRoomMaxDiffHumidity(self):
        return self.maxDiff(1)

    # Print final stats of floor
    def getFinalStats(self):
        print("After " + str(int(self.visits)) + " room visits:")

        # Open created text file to append output
        f = open("heatmiser_trial_output", "a")
        f.write("Final room states: \n")

        # Print + output final room states
        for i in range(12):
            print("Room " + str(i+1) + " -> " + str("%.1f" % self.floor.rooms[i].getTemp()) + "°F & " +
                 str("%.1f" % self.floor.rooms[i].getHumidity()) + "%")
            f.write("Room " + str(i+1) + " -> " + str("%.1f" % self.floor.rooms[i].getTemp()) + "°F & " +
                 str("%.1f" % self.floor.rooms[i].getHumidity()) + "%" + "\n")

        print("")
        print("Average floor temp -> " + str("%.2f" % self.floor.getAverageTemp()) + "°F (" +
                str("%.1f" % self.floor.getStandardDeviationTemp()) + " standard deviations)")
        print("Average floor humidity    -> " + str("%.2f" % self.floor.getAverageHumidity()) + "% (" +
                str("%.2f" % self.floor.getStandardDeviationHumidity()) + " standard deviations)")

        print("<----- END OF TRIAL " + str(self.trial) + " ----->")
        f.write("<----- END OF TRIAL " + str(self.trial) + " -----> \n")
        print("")
        f.close() # close file

    def getVisits(self):
        return self.visits

    def getEnergyUse(self):
        return self.energyUse

    # Check if floor humidity at acceptable average of 47
    def floorHumidityStable(self):
        if (round(self.floor.getAverageHumidity(), 2) >= 47.0) and (round(self.floor.getAverageHumidity(), 2) <= 47.9):
            return True
        return False

    # Check if floor temp at acceptable temp of 72
    def floorTempStable(self):
        if (round(self.floor.getAverageTemp(), 2) >= 72.0) and (round(self.floor.getAverageTemp(), 2) <= 72.9):
            return True
        return False

    # Check if floor humidity standard deviation acceptable <= 1.75
    def floorStandardDeviationHumidityStable(self):
        if (self.floor.getStandardDeviationHumidity() <= 1.75):
            return True
        return False

    # Check if floor temp standard deviation at acceptable <= 1.5
    def floorStandardDeviationTempStable(self):
        if (self.floor.getStandardDeviationTemp() <= 1.5):
            return True
        return False

    # Sets humidity of room to desired percentage
    def updateHumidity(self, roomIndex, newHumidity):
        print("HeatMiser is setting the humidity of room " + str(roomIndex+1) + " to " + str("%.1f" % newHumidity))
        
        # Have floor update room
        self.floor.setRoomHumidity(roomIndex, newHumidity)
        self.floor.calculateStandardDeviation() # ADDED

        # Print updated state of room's humidity
        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % newHumidity) +
              "% humidity, the floor average humidity becomes " + str("%.1f" % self.floor.getAverageHumidity()) +
              "% with an average standard deviation of " + str("%.1f" % self.floor.getStandardDeviationHumidity() + "."))
        print("This room is " + str("%.2f" % (self.floor.getAverageHumidity() - newHumidity)) + " deviations away from the average humidity.")


    # Updates temperature of room to desired degree
    def updateTemp(self, roomIndex, newTemp):
        print("HeatMiser is setting the temp of room " + str(roomIndex+1) + " to " + str("%.1f" % newTemp))
        # Have floor update room
        self.floor.setRoomTemp(roomIndex, newTemp)
        self.floor.calculateStandardDeviation()

        # Print updated state of room's temp
        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % newTemp) + "°F, the floor average temperature becomes " +
              str("%.1f" % self.floor.getAverageTemp()) + "°F with an average standard deviation of " +
              str("%.1f" % self.floor.getStandardDeviationTemp() + "."))
        # print("This is " + str("%.2f" % self.getTempStandardDeviation()) + "x the standard deviation of 1.5")
        print("Room " + str(roomIndex+1) + " is " + str("%.2f" % (self.floor.getAverageTemp() - newTemp)) + " deviations away from the average temp.")


    # Set temperature or humidity of room accordingly
    def chooseAction(self, roomIndex, raiseTemp):
        if raiseTemp is True:
            self.updateTemp(roomIndex, 72)
        else:
            self.updateHumidity(roomIndex, 47)

    # Determines if humidity of room is within accepted range of 45 - 55%, deviations, and average
    def canChangeHumidity(self, currHumidity):
        # Don't touch room if at ideal average humidity
        if ((round(currHumidity, 2) >= 47.0) and (round(currHumidity, 2) <= 47.9)):
            return False
        # Determine if needs to be increased
        elif ((round(currHumidity, 2) < 47.0)):
            self.raiseHumidity = True
        # Determine if humidity needs to be decreased
        elif (round(currHumidity, 2 > 47.9)):
            self.raiseHumidity = False

        # Checks if average is acceptable
        if not self.floorHumidityStable():
            # Checks lower bound
            if not self.raiseHumidity and (currHumidity - 1) >= 45:
                return True
            # Checks upper bound
            elif self.raiseHumidity and (currHumidity + 1) <= 55:
                return True
        # Checks if standard deviation is acceptable
        elif not self.floorStandardDeviationHumidityStable():
            return True
        return False


    # Determines if temp of room is within accepted range of 65 - 75 F
    def canChangeTemp(self, currTemp):
        # Don't touch room if at ideal average humidity
        if ((round(currTemp, 2) >= 72.0) and (round(currTemp,2) <= 72.9)):
            return False
        # Determine if temp needs to be increased
        elif (round(currTemp, 2) < 72.0):
            self.raiseTemp = True
        # Determine if temp needs to be decreased
        elif (round(currTemp, 2) > 72.9):
            self.raiseTemp = False

        # Checks if average is acceptable
        if not self.floorTempStable():
            # Checks lower bound
            if not self.raiseTemp and (currTemp - 1) >= 65:
                return True
            # Checks upper bound
            elif self.raiseTemp and (currTemp + 1) <= 75:
                return True
        # Checks if standard deviation is acceptable
        elif not self.floorStandardDeviationTempStable():
            return True

        return False

    # Checks if room has a stable standard deviation or not
    def roomStandardDeviationTempStable(self, currTemp):
        if abs(self.floor.getAverageTemp() - currTemp) <= 1.5:
            return True
        return False

    # Main function to drive Optimized HeatMiser
    def unoptimizedRun(self):
        # Initialize HeatMiser at a random room
        roomIndex = 0

        # Run on the rooms of the floor
        while not (self.floorHumidityStable() and self.floorTempStable()):
            print("HeatMiser is in room " + str(roomIndex+1))

            currRoom = self.floor.rooms[roomIndex]
            currHumidity = currRoom.getHumidity()
            currTemp = currRoom.getTemp()

            print("Room " + str(roomIndex+1) + " is at " + str("%.1f" % currTemp) + "°F & " +
                  str("%.1f" % currHumidity) + "% humidity")

            self.updateTemp()
            print("Floor averages: temp: " + str("%.1f" % self.floor.getAverageTemp()) + ", humidity:" + str("%.1f" % self.floor.getAverageHumidity()))

            if roomIndex < 11:
                roomIndex += 1
            else:
                roomIndex = 0

            # Increment total visits
            self.visits += 1
            print("Moving on ----->")
            print("")

        self.getFinalStats()


    # Returns array path to target room using BFS from start index to target index
    def findPathBFS(self, graph, start, target):
        queue = [[start]] # keep track of rooms to be checked using a queue
        explored = {} # keep track of rooms visited

        # continue until target node found
        while queue:
            # Gets first path in the queue
            path = queue.pop(0)
            # Get last node in the path
            node = path[-1]

            # Check if at end, if so return path
            if node == target:
                return path
            # Add to explored and continue
            elif node not in explored:
                # Construct a new path
                for neighbor in graph[node]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                explored[node] = True
        
        return path
    
    # Heatmiser searches for office with largest difference - baseline
    def baselineRun(self):
        # Initialize HeatMiser at a random room
        roomIndex = randrange(0,12)
        graph = self.floor.getFloorPlan()

        # Run on the rooms of the floor
        while not (self.floorHumidityStable() and self.floorTempStable()):
            print("HeatMiser is in room " + str(roomIndex+1))

            # Determine max temp and max humidity difference
            maxRoomDiffTemp = self.getRoomMaxDiffTemperature()
            maxRoomDiffHumidity = self.getRoomMaxDiffHumidity()

            # Go to room with greatest diff first
            # temperature
            if ((max(maxRoomDiffTemp.getTemp(), maxRoomDiffHumidity.getHumidity()) == maxRoomDiffTemp.getTemp())
                and (not self.floorTempStable())):
                targetRoom = maxRoomDiffTemp
                otherRoom = maxRoomDiffHumidity
                tempFirst = True
            else:
                targetRoom = maxRoomDiffHumidity
                otherRoom = maxRoomDiffTemp
                tempFirst = False

            # Update room index to target

            # Get path to room with greatest max
            path = self.findPathBFS(graph, roomIndex+1, targetRoom.getIndex()+1)
            energyCost = self.getFinalCost(path)

            print("HeatMiser detects the max room to be " + str(targetRoom.getIndex()+1) + " at " + str("%.1f" % targetRoom.getTemp()) + "°F & " +
                str("%.1f" % targetRoom.getHumidity()) + "% humidity")
            print("HeatMiser is going to room " + str(targetRoom.getIndex()+1) + ". The path to room " + str(targetRoom.getIndex()+1) + " is:")
            print(path[1:])
            self.chooseAction(targetRoom.getIndex(), tempFirst)

            # Check if other max room en route to target
            if (otherRoom.getIndex() in path):
                print("The other max diff - room " + str(targetRoom.getIndex()+1) + " is on the way!")
                self.chooseAction(otherRoom.getIndex(), not tempFirst)

            print("The total accumulated energy cost of this path was: " + str(energyCost))
            print("Floor averages: temp: " + str("%.1f" % self.floor.getAverageTemp()) + ", humidity:" + str("%.1f" % self.floor.getAverageHumidity()))

            # Increment total visits by number of rooms passed
            self.visits += (len(path) - 1)
            self.energyUse += energyCost
            roomIndex = targetRoom.getIndex()
            print("Moving on ----->")
            print("")

        self.getFinalStats()
        #print(HeuristicParser.getHeuristic())

    # returns heuristic cost
    def getHeuristicCost(self, next, end):
        return self.floor.getHeuristicCost()[next][end]

    # returns energy/edge weight cost
    def getEnergyCost(self, current, next):
        return self.floor.getFloorCost()[current][next]

        # takes the final node from A* and backtracks to return best path
    def recreatePath(self, cameFrom, current):
        path = [current]

        while current in cameFrom.keys():
            current = cameFrom[current]
            path.append(current)

        return list(reversed(path))

    # Given a final path, returns cost associated with it
    def getFinalCost(self, path):
        totalCost = 0

        for i in range(0, len(path)-1):
            totalCost += self.getEnergyCost(path[i], path[i+1])

        return totalCost

    # Returns array path to target room using A* search from start index to target index
    def aStar(self, graph, start, target):
        openSet = PriorityQueue()  # keep track of rooms to be checked using a queue
        closedSet = []  # keep track of rooms visited
        cameFrom = {}
        fromStart = {}
        totalCost = {}
        fromStart[start] = 0

        totalCost[start] = self.getHeuristicCost(start, target)
        openSet.put((totalCost[start], start))

        while openSet:
            current = openSet.get()[1]

            if (current == target):
                return self.recreatePath(cameFrom, current)

            closedSet.append(current)

            for neighbor in graph[current]:
                if (not (neighbor in closedSet)):
                    tentativeEnergyCost = fromStart[current] + self.getEnergyCost(current, neighbor)

                    if ((neighbor not in fromStart) or (tentativeEnergyCost < fromStart[neighbor])):
                        cameFrom[neighbor] = current
                        fromStart[neighbor] = tentativeEnergyCost

                        if (neighbor == target):
                            totalCost[neighbor] = fromStart[neighbor]
                        else:
                            totalCost[neighbor] = fromStart[neighbor] + self.getHeuristicCost(neighbor, target)
                        openSet.put((totalCost[neighbor], neighbor))
        return None

    # Heatmiser searches based on heuristic + weight (A* search)
    def heuristicRun(self):
        # Initialize HeatMiser at a random room
        roomIndex = randrange(0,12)
        graph = self.floor.getFloorPlan()

        # Run on the rooms of the floor
        while not (self.floorHumidityStable() and self.floorTempStable()):
            print("HeatMiser is in room " + str(roomIndex+1))

            # Determine max temp and max humidity difference
            maxRoomDiffTemp = self.getRoomMaxDiffTemperature()
            maxRoomDiffHumidity = self.getRoomMaxDiffHumidity()

            # Go to room with greatest diff first
            # temperature
            if ((max(maxRoomDiffTemp.getTemp(), maxRoomDiffHumidity.getHumidity()) == maxRoomDiffTemp.getTemp())
                and (not self.floorTempStable())):
                targetRoom = maxRoomDiffTemp
                otherRoom = maxRoomDiffHumidity
                tempFirst = True
            else:
                targetRoom = maxRoomDiffHumidity
                otherRoom = maxRoomDiffTemp
                tempFirst = False

            # Update room index to target

            # Get path to room with greatest max
            if (roomIndex != targetRoom.getIndex()):
                path = self.aStar(graph, roomIndex + 1, targetRoom.getIndex() + 1)
                energyCost = self.getFinalCost(path)

                print("HeatMiser detects the max room to be " + str(targetRoom.getIndex() + 1) + " at " + str(
                    "%.1f" % targetRoom.getTemp()) + "°F & " +
                      str("%.1f" % targetRoom.getHumidity()) + "% humidity")
                print("HeatMiser is going to room " + str(targetRoom.getIndex() + 1) + ". The path to room " + str(
                    targetRoom.getIndex() + 1) + " is:")
                print(path[1:])
                print("which uses " + str(energyCost) + " energy.\n")

                # Check if other max room en route to target
                if (otherRoom.getIndex() in path):
                    print("The other max diff - room " + str(targetRoom.getIndex() + 1) + " is on the way!")
                    self.chooseAction(otherRoom.getIndex(), not tempFirst)
                    print("\n")



                self.energyUse += energyCost
                self.visits += (len(path) - 1)
                roomIndex = targetRoom.getIndex()

            else:
                print("HeatMiser detects the max room to be the current room, number" + str(roomIndex+1) +  " at "
                      + str("%.1f" % targetRoom.getTemp()) + "°F & "
                      +  str("%.1f" % targetRoom.getHumidity()) + "% humidity")

                self.visits += 1

            self.chooseAction(targetRoom.getIndex(), tempFirst)

            print("\nFloor averages: temp: " + str("%.1f" % self.floor.getAverageTemp()) + ", humidity: "
                  + str("%.1f" % self.floor.getAverageHumidity()))

            # Increment total visits by number of rooms passed
            print("Moving on ----->")
            print("")

        self.getFinalStats()

def main():
    totalVisits = totalEnergyUsed = 0
    totalTempDeviation = totalHumidityDeviation = 0.0

    # Create text file to write to. Overwrites previous trial
    f = open("optimized_heatmiser_trial_output", "w")
    f.close()

    for i in range(100):
        floor = Floor()
        optimizedHeatMiser = OptimizedHeatMiser(floor, i+1)
        optimizedHeatMiser.baselineRun()
        # optimizedHeatMiser.heuristicRun()
        floor.calculateStandardDeviation()

        totalEnergyUsed += optimizedHeatMiser.getEnergyUse()
        totalVisits += optimizedHeatMiser.getVisits()
        totalTempDeviation += floor.getStandardDeviationTemp()
        totalHumidityDeviation += floor.getStandardDeviationHumidity()

    # final breakdown after 100 trials
    print("The HeatMiser had an average of " + str(int(totalVisits/100)) + " office visits per trial, and had an"
            + " average energy use of " + str(int(totalEnergyUsed/100)) + ".")
    print("It ended, on average, with a final temp standard deviation of " + str("%.1f" % (totalTempDeviation/100))
          + " and a final humidity standard deviation of " + str("%.2f" % (totalHumidityDeviation/100)) + ".")


if __name__ == '__main__':
    main()

