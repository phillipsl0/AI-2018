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
		self.most_edges = []
		self.edges_pointer = len(self.rooms) - 1
		self.already_colored = []
		self.colored_pointer = 0

	def create_edges_stack(self):
		edges = {}

		for room in self.rooms:
			edges[room] = len(self.rooms[room].get_neighbors())

		self.most_edges = sorted(edges, key=edges.__getitem__)

	def pop_edges(self):
		self.colored_pointer += 1
		self.already_colored.append(self.most_edges[self.edges_pointer])

		self.most_edges.pop(self.edges_pointer)
		self.edges_pointer -= 1

	def pop_colored(self):
		self.edges_pointer += 1
		self.most_edges.append(self.already_colored[self.colored_pointer])

		self.already_colored.pop(self.colored_pointer)
		self.colored_pointer -= 1

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
		of2.set_neighbors([wh1, wh2, of3, of4])
		of3.set_neighbors([of2, wh2, wh3, of4])
		of4.set_neighbors([wh1, of2, of3, wh4, of5])
		of5.set_neighbors([wh1, of4, wh4, of6])
		of6.set_neighbors([of1, of5, wh4])
		wh3.set_neighbors([wh2, wh4, of3])
		wh4.set_neighbors([of4, wh3, of6, of5])

		# Add to floor
		self.rooms["Warehouse 1"] =  {"Room": wh1, "Action": None}
		self.rooms["Warehouse 2"] =  {"Room": wh2, "Action": None}
		self.rooms["Warehouse 3"] =  {"Room": wh3, "Action": None}
		self.rooms["Warehouse 4"] =  {"Room": wh4, "Action": None}
		self.rooms["Office 1"] = {"Room": of1, "Action": None}
		self.rooms["Office 2"] = {"Room": of2, "Action": None}
		self.rooms["Office 3"] = {"Room": of3, "Action": None}
		self.rooms["Office 4"] = {"Room": of4, "Action": None}
		self.rooms["Office 5"] = {"Room": of5, "Action": None}
		self.rooms["Office 6"] = {"Room": of6, "Action": None}

	# Returns starting room
	def get_initial_room(self):
		keys = list(self.rooms.keys())
		size = len(keys)
		rand_room_index = randrange(len(keys))
		rand_room_name = keys[rand_room_index]
		return self.rooms[rand_room_name]["Room"]

	# Checks if all rooms have actions
	def check_floor_actions(self):
		for room in self.rooms:
			if self.rooms[room]["Action"] is None:
				return False
		return True

	# Attempt room action
	def attempt_room_action(self, room, action):
		neighbors = room.get_neighbors()
		
		print("In room " + room.get_name() + " - neighbors:")
		print(", ".join([neighbor.get_name() for neighbor in neighbors]))
		print("Current action: " + action)
		for neighbor in neighbors:
			try:
				print("Checking neighbor: " + neighbor.get_name() + " - action: " + neighbor.get_action())
			except:
				print("Checking neighbor: " + neighbor.get_name() + " - action: None")

			# Can't do action - neighbor has similar action
			if (self.rooms[neighbor.get_name()]["Action"] == action):
				print("Actions match!!")
				return False

		return True

	# Set room action
	def set_room_action(self, room, action):
		roomName = room.get_name()

		self.rooms[roomName]["Action"] = action
		room.set_action(action)

	# Get adjacent room with no action, else None
	def next_room(self, room):
		neighbors = room.get_neighbors()

		for neighbor in neighbors:
			if neighbor.get_action() == None:
				return neighbor

		return None

	# Prints actions of floor
	def print_floor_mapping(self):
		for roomName in self.rooms:
			roomAction = self.rooms[roomName]["Room"].get_action()
			if roomAction is None:
				print("Room " + roomName + ": None")
			else:
				print("Room " + roomName + ": " + roomAction)


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
			print("Room stack: ")
			print([neighbor.get_name() for neighbor in currRoom.get_neighbors()])

			action = self.get_room_action(currRoom) # adds action to room history

			# no actions available, backtrack
			if action is None:
				fails += 1
				self.clear_room_history(currRoom)
				currRoom = room_stack.pop()
			else:
				# Check whether action can be done on room
				success = self.floor.attempt_room_action(currRoom, action)
				
				# Action valid - set action and add to stack
				if success:
					self.floor.set_room_action(currRoom, action)
					room_stack.append(currRoom)
					currRoom = self.floor.next_room(currRoom) # get next room with no action
				
				# Test if pass
				if self.floor.check_floor_actions():
					break

				# No valid neighbors - backtrack
				if (currRoom is None):
					currRoom = room_stack.pop() # retrieve last room

			print("***")
			self.floor.print_floor_mapping()
			print("\n")

		print("Final Mapping")
		self.floor.print_floor_mapping()


def main():
	heatMiser = ConstraintOptimalHeatMiser()
	heatMiser.brute_force()
	# heatMiser.floor.create_edges_stack()

if __name__ == '__main__':
	main()
