import random

# Room class that sets and returns its humidity and temperature
class Room:
    def __init__(self, temperature, humidity):
        self.temperature = temperature
        self.humidity = humidity

    def setTemperature(self, temperature):
        self.temperature = temperature

    def setHumidity(self, humidity):
        self.humidity = humidity

    def getTemperature(self):
        return self.temperature

    def getHumidity(self):
        return self.humidity


#
class Floor:
    def __init__(self):
        self.rooms = []
        self.totalTemp = self.totalHumidity = 0
        self.avgTemp = self.avgHumidity = 0.0
        print("Starting room states:")

        # Generates 12 random rooms in the floor
        for i in range(12):
            room = Room(random.uniform(65.0, 75.0), random.uniform(45.0, 55.0))
            # Python random for temperature
            # which cannot exceed the range of 65-75 degrees or 45-55% humidity
            self.totalHumidity += room.getHumidity()
            self.totalTemp += room.getTemperature()
            print("Room " + str(i+1) + ": " + str("%.1f" % room.getTemperature()) + "°F & " +
                  str("%.1f" %room.getHumidity()) + "%")

            self.rooms.append(room)

        print("")
        self.calculateAverageTemp()
        self.calculateAverageHumidity()

    def setRoomTemperature(self, index, temperature):
        oldTemp = self.rooms[index].getTemperature()
        self.rooms[index].setTemperature(temperature)
        self.updateTotalTemperature(oldTemp, temperature)
        self.calculateAverageTemp()

    def setRoomHumidity(self, index, humidity):
        oldHumidity = self.rooms[index].getHumidity()
        self.rooms[index].setHumidity(humidity)
        self.updateTotalHumidity(oldHumidity, humidity)
        self.calculateAverageHumidity()

    def updateTotalTemperature(self, oldTemp, temperature):
        self.totalTemp -= oldTemp
        self.totalTemp += temperature

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


#
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

    def floorTemperatureStable(self):
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

        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % currHumidity) +
              "% humidity, the floor average becomes " + str("%.1f" % self.floor.getAverageHumidity()) + "%.")
        print("This is " + str("%.2f" % self.getHumidityStandardDeviation()) + "x the standard deviation of 1.75")

    def updateTemperature(self, currTemp, roomIndex, currRoom):
        # Decide to either increase or decrease humidity
        if self.raiseTemp:
            print("HeatMiser is raising the temperature of room " + str(roomIndex+1) + " by 1°F")
            currTemp += 1
        elif (not self.raiseTemp):
            print("HeatMiser is lowering the temperature of room " + str(roomIndex+1) + " by 1°F")
            currTemp -= 1

        # Have floor update room
        self.floor.setRoomTemperature(roomIndex, currTemp)

        print("With room " + str(roomIndex+1) + " now at " + str("%.1f" % currTemp) + "°F, the floor average becomes " +
              str("%.1f" % self.floor.getAverageTemp()) + "°F.")
        print("This is " + str("%.2f" % self.getTempStandardDeviation()) + "x the standard deviation of 1.5")

    def getFinalStats(self):
        print("After " + str(int(self.visits)) + " room visits:")

        for i in range(12):
            print("Room " + str(i+1) + " -> " + str("%.1f" % self.floor.rooms[i].getTemperature()) + "°F & " +
                 str("%.1f" % self.floor.rooms[i].getHumidity()) + "%")

        print("")

        print("Average floor temperature -> " + str("%.2f" % self.floor.getAverageTemp()) + "°F (" +
              str("%.2f" % self.getTempStandardDeviation()) + "x std. dev)")
        print("Average floor humidity    -> " + str("%.2f" % self.floor.getAverageHumidity()) + "% (" +
              str("%.2f" % self.getHumidityStandardDeviation()) + "x std. dev)")

        print("<----- END OF TRIAL " + str(self.trial) + " ----->")
        print("")

    def getVisits(self):
        return self.visits

    def chooseAction(self, currRoom, roomIndex, currHumidity, currTemp):
        action = random.randrange(0,2)

        if action == 0:
            # Change humidity if not comfortable
            if self.canChangeHumidity(currHumidity):
                self.updateHumidity(currHumidity, roomIndex, currRoom)
            elif self.canChangeTemperature(currTemp):
                self.updateTemperature(currTemp, roomIndex, currRoom)
            else:
                print("HeatMiser has chosen to do nothing in this room.")

        else:
            if self.canChangeTemperature(currTemp):
                self.updateTemperature(currTemp, roomIndex, currRoom)
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

    def canChangeTemperature(self, currTemp):
        if not self.floorTemperatureStable():
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

        while not (self.floorHumidityStable() and self.floorTemperatureStable()):
            print("HeatMiser is in room " + str(roomIndex+1))

            currRoom = self.floor.rooms[roomIndex]
            currHumidity = currRoom.getHumidity()
            currTemp = currRoom.getTemperature()

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
    totalTemperatureDeviation = totalHumidityDeviation = 0.0

    for i in range(100):
        floor = Floor()
        heatMiser = HeatMiser(floor, i+1)
        heatMiser.run()

        totalVisits += heatMiser.getVisits()
        totalTemperatureDeviation += heatMiser.getTempStandardDeviation()
        totalHumidityDeviation += heatMiser.getHumidityStandardDeviation()

    # final breakdown after 100 trials
    print("The HeatMiser had an average of " + str(int(totalVisits/100)) + " office visits per trial,")
    print("ending, on average, with a final temperature " + str("%.2f" % (totalTemperatureDeviation/100)) +
          "x the standard deviation,")
    print("and a final humidity " + str("%.2f" % (totalHumidityDeviation/100)) + "x the standard deviation,")


if __name__ == '__main__':
    main()