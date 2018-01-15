import random

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
        self.avgTemp = self.avgHumidity = self.avgStandardDeviation = 0.0
        print("Starting room states:")

        # Open created text file to append output
        f = open("heatmiser_trial_output", "a")
        f.write("Starting room states: \n")

        # Generates 12 random rooms in the floor
        for i in range(12):
            # Python random for temp
            # which cannot exceed the range of 65-75 degrees or 45-55% humidity
            room = Room(random.uniform(65.0, 75.0), random.uniform(45.0, 55.0))
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

    def calculateStandardDeviation(self):
        mean 


# HeatMiser class that adjusts humidity and temp to comfortable levels
class HeatMiser:
    def __init__(self, floor, trial):
        self.floor = floor
        self.raiseTemp = self.raiseHumidity = None
        self.visits = 0
        self.trial = trial

    def floorHumidityStable(self):
        if (self.floor.getAverageHumidity() >= 45.25) and (self.floor.getAverageHumidity() <= 48.75):
            return True
        return False

    def floorTempStable(self):
        if (self.floor.getAverageTemp() >= 70.5) and (self.floor.getAverageTemp() <= 73.5):
            return True
        return False

    def getTempStandardDeviation(self):
        if self.raiseTemp:
            return (72-self.floor.getAverageTemp())/1.5
        else:
            return (self.floor.getAverageTemp()-72)/1.5

    def getHumidityStandardDeviation(self):
        if self.raiseHumidity:
            return (self.floor.getAverageHumidity() - 47)/ 1.75

        else:
            return (self.floor.getAverageHumidity() - 47)/1.75

    # Checks states of room humidity and raises or decreases accordingly
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

        # Print updated state of room's humidity
        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % currHumidity) +
              "% humidity, the floor average becomes " + str("%.1f" % self.floor.getAverageHumidity()) + "%.")
        print("This is " + str("%.2f" % self.getHumidityStandardDeviation()) + "x the standard deviation of 1.75")

    # Checks states of room temp and raises or decreases accordingly
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

        # Print updated state of room's temp
        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % currTemp) + "°F, the floor average becomes " +
              str("%.1f" % self.floor.getAverageTemp()) + "°F.")
        print("This is " + str("%.2f" % self.getTempStandardDeviation()) + "x the standard deviation of 1.5")

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
              str("%.2f" % self.getTempStandardDeviation()) + "x std. dev)")
        print("Average floor humidity    -> " + str("%.2f" % self.floor.getAverageHumidity()) + "% (" +
              str("%.2f" % self.getHumidityStandardDeviation()) + "x std. dev)")

        print("<----- END OF TRIAL " + str(self.trial) + " ----->")
        f.write("<----- END OF TRIAL " + str(self.trial) + " -----> \n")
        print("")
        f.close() # close file

    def getVisits(self):
        return self.visits

    def chooseAction(self, currRoom, roomIndex, currHumidity, currTemp):
        # Randomized whether to check humidity or temp first
        action = random.randrange(0,2)

        if action == 0:
            # Change humidity if not comfortable
            if self.canChangeHumidity(currHumidity):
                self.updateHumidity(currHumidity, roomIndex, currRoom)
            elif self.canChangeTemp(currTemp):
                self.updateTemp(currTemp, roomIndex, currRoom)
            else:
                print("HeatMiser has chosen to do nothing in this room.")

        else:
            # Change temp if not comfortable
            if self.canChangeTemp(currTemp):
                self.updateTemp(currTemp, roomIndex, currRoom)
            elif self.canChangeHumidity(currHumidity):
                self.updateHumidity(currHumidity, roomIndex, currRoom)
            else:
                print("HeatMiser has chosen to do nothing in this room.")

    def canChangeHumidity(self, currHumidity):
        if not self.floorHumidityStable():
            if not self.raiseHumidity and (currHumidity - 1) >= 45:
                return True
            elif self.raiseHumidity and (currHumidity + 1) <= 55:
                return True
        return False

    def canChangeTemp(self, currTemp):
        if not self.floorTempStable():
            if not self.raiseTemp and (currTemp - 1) >= 65:
                return True
            elif self.raiseTemp and (currTemp + 1) <= 75:
                return True
        return False

    # Main function to drive HeatMiser
    def run(self):
        roomIndex = 0

        # Determine whether to increase or decrease temp and or humidity
        if self.floor.getAverageHumidity() < 45.25:
            self.raiseHumidity = True
        else:
            self.raiseHumidity = False

        if self.floor.getAverageTemp() < 70.5:
            self.raiseTemp = True
        else:
            self.raiseTemp = False

        # Run on the rooms of the floor
        while not (self.floorHumidityStable() and self.floorTempStable()):
            print("HeatMiser is in room " + str(roomIndex+1))

            currRoom = self.floor.rooms[roomIndex]
            currHumidity = currRoom.getHumidity()
            currTemp = currRoom.getTemp()

            print("Room " + str(roomIndex+1) + " is at " + str("%.1f" % currTemp) + "°F & " +
                  str("%.1f" % currHumidity) + "% humidity")

            self.chooseAction(currRoom, roomIndex, currHumidity, currTemp)

            if roomIndex < 11:
                roomIndex += 1
            else:
                roomIndex = 0

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

        totalVisits += heatMiser.getVisits()
        totalTempDeviation += heatMiser.getTempStandardDeviation()
        totalHumidityDeviation += heatMiser.getHumidityStandardDeviation()

    # final breakdown after 100 trials
    print("The HeatMiser had an average of " + str(int(totalVisits/100)) + " office visits per trial,")
    print("ending, on average, with a final temp " + str("%.2f" % (totalTempDeviation/100)) +
          "x the standard deviation,")
    print("and a final humidity " + str("%.2f" % (totalHumidityDeviation/100)) + "x the standard deviation,")


if __name__ == '__main__':
    main()