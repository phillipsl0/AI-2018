from random import uniform, randrange
from math import sqrt

# Room class that sets and returns its humidity and temp
class Room:
    def __init__(self, temp, humidity):
        self.temp = temp
        self.humidity = humidity

    def setTemp(self, temp):
        self.temp = temp

    def setHumidity(self, humidity):
        self.humidity = humidity

    def getTemp(self):
        return self.temp

    def getHumidity(self):
        return self.humidity


# Floor class that generates rooms and sets average humidity and temp
class Floor:
    def __init__(self):
        self.rooms = []
        self.totalTemp = self.totalHumidity = 0
        self.avgTemp = self.avgHumidity = self.stdTemp = self.stdHumidity = 0.0
        print("Starting room states:")

        # Open created text file to append output
        f = open("heatmiser_trial_output", "a")
        f.write("Starting room states: \n")

        # Generates 12 random rooms in the floor
        for i in range(12):
            # Python random for temp
            # which cannot exceed the range of 65-75 degrees or 45-55% humidity
            
            room = Room(uniform(65.0, 75.0), uniform(45.0, 55.0))
        
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

    def calculateAverageTemp(self):
        self.avgTemp = self.totalTemp / 12

    def getAverageTemp(self):
        return self.avgTemp

    def calculateAverageHumidity(self):
        self.avgHumidity = self.totalHumidity / 12

    def getAverageHumidity(self):
        return self.avgHumidity

    # Calculates standard deviation for both humidity and temperature of floor
    def calculateStandardDeviation(self):
        xTemp = 0
        xHumidity = 0

        for room in self.rooms:
            xTemp += ((room.getTemp() - self.avgTemp)**2)
            xHumidity += ((room.getHumidity() - self.avgHumidity)**2)

        self.stdTemp = sqrt(xTemp / 12)
        self.stdHumidity = sqrt(xHumidity / 12)

    def getStandardDeviationTemp(self):
        return self.stdTemp

    def getStandardDeviationHumidity(self):
        return self.stdHumidity


# HeatMiser class that adjusts humidity and temp to comfortable levels
class HeatMiser:
    def __init__(self, floor, trial):
        self.floor = floor
        self.raiseTemp = self.raiseHumidity = None
        self.visits = 0
        self.trial = trial

    # Check if floor humidity at acceptable average of 47
    def floorHumidityStable(self):
        if (round(self.floor.getAverageHumidity(), 2) >= 47.0) and (round(self.floor.getAverageHumidity(), 2) <= 47.9):
            return True
        return False

    # Check if floor temperature at acceptable temperature of 72
    def floorTempStable(self):
        if (round(self.floor.getAverageTemp(), 2) >= 72.0) and (round(self.floor.getAverageTemp(), 2) <= 72.9):
            return True
        return False

    # Check if floor humidity standard deviation acceptable <= 1.75
    def floorStandardDeviationHumidityStable(self):
        if (self.floor.getStandardDeviationHumidity() <= 1.75):
            return True
        return False

    # Check if floor temperature standard deviation at acceptable <= 1.5
    def floorStandardDeviationTemperatureStable(self):
        if (self.floor.getStandardDeviationTemp() <= 1.5):
            return True
        return False

    # Checks states of room humidity and raises or decreases accordingly. Updates standard deviation.
    def updateHumidity(self, currHumidity, roomIndex, currRoom):
        # Decide to either increase or decrease humidity
        if self.raiseHumidity:
            print("HeatMiser is raising the humidity of room " + str(roomIndex+1) + " by 1%")
            currHumidity += 1
        elif (not self.raiseHumidity):
            print("HeatMiser is lowering the humidity of room " + str(roomIndex+1) + " by 1%")
            currHumidity -= 1

        # Have floor update room
        self.floor.setRoomHumidity(roomIndex, currHumidity)
        self.floor.calculateStandardDeviation() # ADDED

        # Print updated state of room's humidity
        # CHANGED
        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % currHumidity) +
              "% humidity, the floor average becomes " + str("%.1f" % self.floor.getAverageHumidity()) +
              "% with an average standard deviation of " + str("%.1f" % self.floor.getStandardDeviationHumidity() + "."))
        # print("This is " + str("%.2f" % self.getHumidityStandardDeviation()) + "x the standard deviation of 1.75")
        print("This room is " + str("%.2f" % (self.floor.getAverageHumidity() - currHumidity)) + " deviations away from the average humidity.")


    # Checks states of room temp and raises or decreases accordingly. Updates standard deviation.
    def updateTemp(self, currTemp, roomIndex, currRoom):
        # Decide to either increase or decrease humidity
        if self.raiseTemp:
            print("HeatMiser is raising the temp of room " + str(roomIndex+1) + " by 1°F")
            currTemp += 1
        elif (not self.raiseTemp):
            print("HeatMiser is lowering the temp of room " + str(roomIndex+1) + " by 1°F")
            currTemp -= 1

        # Have floor update room
        self.floor.setRoomTemp(roomIndex, currTemp)
        self.floor.calculateStandardDeviation() # ADDED

        # Print updated state of room's temp
        # CHANGED
        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % currTemp) + "°F, the floor average becomes " +
              str("%.1f" % self.floor.getAverageTemp()) + "°F with an average standard deviation of " +
              str("%.1f" % self.floor.getStandardDeviationTemp() + "."))
        # print("This is " + str("%.2f" % self.getTempStandardDeviation()) + "x the standard deviation of 1.5")
        print("This is room is " + str("%.2f" % (self.floor.getAverageTemp() - currTemp)) + " deviations away from the average temperature.")

        
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

        # CHANGED
        # print("Average floor temp -> " + str("%.2f" % self.floor.getAverageTemp()) + "°F (" +
        #       str("%.2f" % self.getTempStandardDeviation()) + "x std. dev)")
        # print("Average floor humidity    -> " + str("%.2f" % self.floor.getAverageHumidity()) + "% (" +
        #       str("%.2f" % self.getHumidityStandardDeviation()) + "x std. dev)")
        print("Average floor temp -> " + str("%.2f" % self.floor.getAverageTemp()) + "°F (" +
                str("%.2f" % self.floor.getStandardDeviationTemp()) + " standard deviations)")
        print("Average floor humidity    -> " + str("%.2f" % self.floor.getAverageHumidity()) + "% (" +
                str("%.2f" % self.floor.getStandardDeviationHumidity()) + " standard deviations)")

        print("<----- END OF TRIAL " + str(self.trial) + " ----->")
        f.write("<----- END OF TRIAL " + str(self.trial) + " -----> \n")
        print("")
        f.close() # close file

    def getVisits(self):
        return self.visits

    def chooseAction(self, currRoom, roomIndex, currHumidity, currTemp):
        # Randomized whether to check humidity or temp first
        action = randrange(0,2)

        # Looks at humidity first
        if action == 0:
            if self.canChangeHumidity(currHumidity):
                self.updateHumidity(currHumidity, roomIndex, currRoom)
            elif self.canChangeTemp(currTemp):
                self.updateTemp(currTemp, roomIndex, currRoom)
            else:
                print("HeatMiser has chosen to do nothing in this room.")

        # Looks at temperature first
        else:
            if self.canChangeTemp(currTemp):
                self.updateTemp(currTemp, roomIndex, currRoom)
            elif self.canChangeHumidity(currHumidity):
                self.updateHumidity(currHumidity, roomIndex, currRoom)
            else:
                print("HeatMiser has chosen to do nothing in this room.")

    # Determines if humidity of room is within accepted range of 45 - 55%, deviations, and average
    def canChangeHumidity(self, currHumidity):
        # Don't touch room if at ideal average humidity
        if ((round(currHumidity, 2) >= 47.0) and (round(currHumidity, 2) <= 47.9)):
            return False
        # CHANGED
        # Determine if needs to be increased
        elif ((round(currHumidity, 2) < 45.25)):
            self.raiseHumidity = True
        # Determine if humidity needs to be decreased
        elif (round(currHumidity, 2 > 48.75)):
            self.raiseHumidity = False

        if not self.floorHumidityStable():
            # Checks lower bound
            if not self.raiseHumidity and (currHumidity - 1) >= 45:
                return True
            # Checks upper bound
            elif self.raiseHumidity and (currHumidity + 1) <= 55:
                return True
        return False

    # Determines if temperature of room is within accepted range of 65 - 75 F
    def canChangeTemp(self, currTemp):
        # Don't touch room if at ideal average humidity
        if ((round(currTemp, 2) >= 72.0) and (round(currTemp,2) <= 72.9)):
            return False
        # Determine if temp needs to be increased
        elif (round(currTemp, 2) < 70.5):
            self.raiseTemp = True
        # Determine if temp needs to be decreased
        elif (round(currTemp, 2) > 73.5):
            self.raiseTemp = False


        # Checks if temperature needs to be increased
        if not self.floorTempStable():
            # Checks lower bound
            if not self.raiseTemp and (currTemp - 1) >= 65:
                return True
            # Checks upper bound
            elif self.raiseTemp and (currTemp + 1) <= 75:
                return True
        return False

    # Checks if room has a stable standard deviation or not
    def roomStandardDeviationTempStable(self, currTemp):
        if abs(self.floor.getAverageTemp() - currTemp) <= 1.5:
            return True
        return False


    # Main function to drive HeatMiser
    def run(self):
        roomIndex = 0

        # Determine whether to initially increase or decrease temp and or humidity
        # CHANGEDs
        # if self.floor.getAverageHumidity() < 47:
        #     self.raiseHumidity = True
        # else:
        #     self.raiseHumidity = False

        # if self.floor.getAverageTemp() < 72:
        #     self.raiseTemp = True
        # else:
        #     self.raiseTemp = False

        # Run on the rooms of the floor
        while not (self.floorHumidityStable() and self.floorTempStable()):
            print("HeatMiser is in room " + str(roomIndex+1))

            currRoom = self.floor.rooms[roomIndex]
            currHumidity = currRoom.getHumidity()
            currTemp = currRoom.getTemp()

            print("Room " + str(roomIndex+1) + " is at " + str("%.1f" % currTemp) + "°F & " +
                  str("%.1f" % currHumidity) + "% humidity")

            self.chooseAction(currRoom, roomIndex, currHumidity, currTemp)
            print("Floor averages: temperature: " + str("%.1f" % self.floor.getAverageTemp()) + ", humidity:" + str("%.1f" % self.floor.getAverageHumidity()))

            if roomIndex < 11:
                roomIndex += 1
            else:
                roomIndex = 0

            # Increment total visits
            self.visits += 1
            print("Moving on ----->")
            print("")

        self.getFinalStats()


def main():
    totalVisits = 0
    totalTempDeviation = totalHumidityDeviation = 0.0

    # Create text file to write to. Overwrites previous trial
    f = open("heatmiser_trial_output", "w")
    f.close()

    for i in range(100):
        floor = Floor()
        heatMiser = HeatMiser(floor, i+1)
        heatMiser.run()
        floor.calculateStandardDeviation() # ADDED 

        totalVisits += heatMiser.getVisits()
        # CHANGED
        # totalTempDeviation += heatMiser.getTempStandardDeviation()
        # totalHumidityDeviation += heatMiser.getHumidityStandardDeviation()
        totalTempDeviation += floor.getStandardDeviationTemp()
        totalHumidityDeviation += floor.getStandardDeviationHumidity()

    # final breakdown after 100 trials
    # CHANGED
    # print("The HeatMiser had an average of " + str(int(totalVisits/100)) + " office visits per trial,")
    # print("ending, on average, with a final temp " + str("%.2f" % (totalTempDeviation/100)) +
    #       "x the standard deviation,")
    # print("and a final humidity " + str("%.2f" % (totalHumidityDeviation/100)) + "x the standard deviation,")
    print("The HeatMiser had an average of " + str(int(totalVisits/100)) + " office visits per trial,")
    print(" ending, on average, with a final temperature standard deviation of " + str("%.2f" % (totalTempDeviation/100)) +
          " and a final humidity standard deviation of " + str("%.2f" % (totalHumidityDeviation/100)) + ".")


if __name__ == '__main__':
    main()

