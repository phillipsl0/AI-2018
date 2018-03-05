'''
ConstraintOptimalHeatMiser.py
Assignment #4
'''
from random import randrange

# Inner room class that represents a single room
class Room:
	def __init__(self, name):
		self.name = name
		self.action = None
		self.neighbors = []

	def set_neighbors(self, neighbors):
		self.neighbors = neighbors

	def get_neighbors(self):
		return self.neighbors

	def get_action(self):
		return self.action

# Floor class that represents space of heat miser
class Floor:
	def __init__(self):
		self.rooms = {}
		self.create_floor()

	def create_floor(self):
		# Create all the rooms
		wh1 = Room("Warehouse 1")
		wh2 = Room("Warehouse 2")
		wh3 = Room("Warehouse 3")
		wh4 = Room("Warehouse 4")
		of1 = Room("Office 1")
		of2 = Room("Office 2")
		of3 = Room("Office 3")
		of4 = Room("Office 4")
		of5 = Room("Office 5")
		of6 = Room("Office 6")

		# Set neighbors
		wh1.set_neighbors([of1, wh2, of2, of4, of5])
		of1.set_neighbors([wh1, of6])
		wh2.set_neighbors([wh1, wh3, of3, of2])
		of2.set_neighbors([wh1, wh2, of4, of4])
		of3.set_neighbors([of2, wh2, wh3, of4])
		of4.set_neighbors([wh1, of2, of3, wh4, of5])
		of5.set_neighbors([wh1, of4, wh4, of6])
		of6.set_neighbors([of1, of5, wh4])
		wh3.set_neighbors([wh2, wh4, of3])
		wh4.set_neighbors([of4, wh3, of6, of5])

		# Add to floor
		self.rooms["Warehouse 1"] = wh1
		self.rooms["Warehouse 2"] = wh2
		self.rooms["Warehouse 3"] = wh3
		self.rooms["Warehouse 4"] = wh4
		self.rooms["Office 1"] = of1
		self.rooms["Office 2"] = of2
		self.rooms["Office 3"] = of3
		self.rooms["Office 4"] = of4
		self.rooms["Office 5"] = of5
		self.rooms["Office 6"] = of6

	# Returns starting room
	def get_initial_room(self):
		keys = self.rooms.keys()
		size = len(keys)
		return self.rooms[keys[randrange(size)]]

# Heat Miser class
class ConstraintOptimalHeatMiser:
	def __init__(self):
		self.floor = Floor()
		self.mapping = {}

	def brute_force(self):
		pass


def main():
	heatMiser = ConstraintOptimalHeatMiser()

if __name__ == '__main__':
	main()
