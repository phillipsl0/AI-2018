
# Room class that sets and returns its humidity and temperature
class Room:
	def __init__(self, temperature, humidity):
		self.temperature = temperature
		self.humidity = humidity

	def setTemperature(self, temperature):
		self.temperature = temperature

	def setHumidity(self, humidity):
		self.humidity = humidity

	def getTemperature(self, temperature):
		return self.temperature

	def getHumidity(self, humidity):
		return self.humidity

# 
class Floor:
	def __init__(self):
		self.rooms = []
		self.totalTemp = self.totalHumidity = 0
		self.avgTemp = self.avgHumidity = 0

		# Generates 12 random rooms in the floor
		for i in range(12):
			room = Room()
			# Python random for temperature
			# which cannot exceed the range of 65-75 degrees or 45-55% humidity
			self.totalHumidity += room.getHumidity()
			self.totalTemp += room.getTemperature()

			self.rooms.insert(room)

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
		self.avgTemp = self.totalTemp/12

	def getAverageTemp(self):
		return self.avgTemp

	def calculateAverageHumidity(self):
		self.avgHumidity = self.totalHumidity/12	

	def getAverageHumidity(self):
		return self.avgHumidity


# 
class HeatMiser:
	def __init__(self, floor):
		self.floor = floor
		self.raiseTemp = None
		self.raiseHumidity = None

	
	def checkFloorHumidity(self):
		if (self.floor.getAverageHumidity() >= 45.25) and (self.floor.getAverageHumidity() <= 48.75):
			return True
		return False

	def checkFloorTemp(self):
		if (self.floor.getAverageTemp() >= 70.5) and (self.floor.getAverageTemp() <= 73.5):
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


		while not (checkFloorHumidity() and checkFloorTemp()):
			currRoom = self.floor[roomIndex]
			# Change humidity if not comfortable
			if not checkFloorHumidity():
				currHumidity = currRoom.getHumidity()

				# Decide to either increase or decrease humidity
				if self.raiseHumidity and (currHumidity < 55):
					currHumidity += 1
				elif (not self.raiseHumidity) and (currHumidity > 45):
					currHumidity -= 1

				# Have floor update room
				self.floor.setRoomHumidity(currRoom, currHumidity)

			# Change temperature if not comfortable
			# Change this - thank you!!
			if not checkFloorTemp():
				currHumidity = currRoom.getHumidity()

				# Decide to either increase or decrease humidity
				if self.raiseHumidity and (currHumidity < 55):
					currHumidity += 1
				elif (not self.raiseHumidity) and (currHumidity > 45):
					currHumidity -= 1

				# Have floor update room
				self.floor.setRoomHumidity(currRoom, currHumidity)

			if roomIndex < 11:
				roomIndex += 1
			else:
				roomIndex = 0

	#anytime agent changes -> check & if within range, set booleans to true
	#once either is in range, stop caring abt it




def main():
	floor = Floor()
	heatMiser = HeatMiser(floor)




if __name__ == '__main__':
	main()




#(6) After making a change to an individual office's temperature or humidity (which cannot exceed the range of 65-75
# degrees or 45-55% humidity), HeatMiser can recalculate the simulation floor averages and associated standard
# deviations.

#(7) HeatMiser can revisit the offices as many times as is required to fix the simulation floor average temperature to
# 72 degrees and the average humidity to 47% with appropriate standard deviations, but HeatMiser must do so in order.

#(8) Once HeatMiser makes a change to an individual office's temperature or humidity, the temperature or humidity is
# fixed (the occupant will not be competing against HeatMiser).

#(9) HeatMiser's effectiveness in the simulation will be measured by the average number of office visits required to
# bring the 12 office simulation floor into cost-effective compliance from 100 trials (each trail will randomly
# initialize the temperatures and humidities of the 12 rooms).

#Code (50 Points):

#For each run of the simulation, your code must output the following to the user:

#The initial random state of the 12 offices

#The office number, temperature, humidity, HeatMiser's decision/change and subsequent recalculation of floor average
# and standard deviation for each office visit

#Once HeatMiser stops (when appropriate average and standard deviation is achieved), the final temperature and humidity
# of the 12 offices, the final averages and standard deviations and the total number of visits for that simulation

#Once all 100 simulations have run, the average number of visits and standard deviation should be output to the user
,
#Please test your code and provide a README.txt with instructions on how to run your code. You will lose points if your
# code fails to run.