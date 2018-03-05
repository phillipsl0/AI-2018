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

	def set_action(self, action):
		self.action = action

	def get_name(self):
		return self.name

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
		self.rooms["Warehouse 1"]["Room"] = wh1
		self.rooms["Warehouse 2"]["Room"] = wh2
		self.rooms["Warehouse 3"]["Room"] = wh3
		self.rooms["Warehouse 4"]["Room"] = wh4
		self.rooms["Office 1"]["Room"] = of1
		self.rooms["Office 2"]["Room"] = of2
		self.rooms["Office 3"]["Room"] = of3
		self.rooms["Office 4"]["Room"] = of4
		self.rooms["Office 5"]["Room"] = of5
		self.rooms["Office 6"]["Room"] = of6

		self.rooms["Warehouse 1"]["Action"] = None
		self.rooms["Warehouse 2"]["Action"] = None
		self.rooms["Warehouse 3"]["Action"] = None
		self.rooms["Warehouse 4"]["Action"] = None
		self.rooms["Office 1"]["Action"] = None
		self.rooms["Office 2"]["Action"] = None
		self.rooms["Office 3"]["Action"] = None
		self.rooms["Office 4"]["Action"] = None
		self.rooms["Office 5"]["Action"] = None
		self.rooms["Office 6"]["Action"] = None

	# Returns starting room
	def get_initial_room(self):
		keys = self.rooms.keys()
		size = len(keys)
		return self.rooms[keys[randrange(size)]]

	# Checks if all rooms have actions
	def check_floor_actions(self):
		for room in self.rooms:
			if self.rooms[room]["Action"] is None:
				return False
		return True

	# Attempt room action
	def attempt_room_action(self, room, action):
		neighbors = room.get_neighbors()
		
		for neighbor in neighbors:
			# Can't do action - neighbor has similar action
			if self.rooms[room.get_name()]["Action"] == action:
				return False

		return True

	# Set room action
	def set_room_action(room, action):
		roomName = room.get_name()

		self.rooms[roomName]["Action"] = action
		room.set_action(action)


# Heat Miser class
class ConstraintOptimalHeatMiser:
	def __init__(self):
		self.floor = Floor()
		self.actionHistory = {}

	def get_room_action(self, room):
		actions = ["Temp", "Humidity", "Pass"]
		roomName = room.get_name()

		if roomName not in self.actionHistory:
			self.actionHistory[roomName] = []
		history = self.actionHistory[roomName]

		# Get first action not used
		for action in actions:
			if action not in history:
				# add action to heat miser's room history
				history.append(action)
				self.actionHistory[roomName] = history

				return action
		return None

	# Clears current actions done on a room
	def clear_room_history(self, room):
		self.actionHistory[room.get_name()] = []



	def brute_force(self):
		currRoom = self.floor.get_initial_room()
		room_stack = []
		fails = 0 # keeps track of how often heat miser has to backtrack

		while not self.floor.check_floor_actions():
			action = self.get_room_action(currRoom) # adds action to room history

			# back track
			if action is None:
				fail += 1
				self.clear_room_history(currRoom)
				currRoom = room_stack.pop()
			else:
				# Check whether action can be done on room
				success = self.floor.attempt_room_action(currRoom, action)
				
				# Action valid - set action and add to stack
				if success:
					self.floor.set_room_action(currRoom, action)
					room_stack.append(currRoom)
					currRoom = self.floor.next_room()
				else:






def main():
	heatMiser = ConstraintOptimalHeatMiser()

if __name__ == '__main__':
	main()
