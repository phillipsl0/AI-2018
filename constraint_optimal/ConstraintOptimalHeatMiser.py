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
		self.edges_pointer = len(self.rooms) - 1

		for room in self.rooms:
			edges[room] = len(self.rooms[room]["Room"].get_neighbors())

		self.most_edges = sorted(edges, key=edges.__getitem__)

	def pop_edges(self):
		room = self.most_edges.pop(self.edges_pointer)
		self.edges_pointer -= 1

		return room

	def push_edges(self, room):
		self.most_edges.append(room)
		self.edges_pointer += 1

	def reset_colored(self):
		self.already_colored = []
		self.colored_pointer = 0

	def pop_colored(self):
		room = self.already_colored.pop(self.colored_pointer)
		self.colored_pointer -= 1

		return room

	def push_colored(self, room):
		self.already_colored.append(room)
		self.colored_pointer += 1

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

	# Returns starting Room object
	def get_initial_room_random(self):
		keys = list(self.rooms.keys())
		size = len(keys)
		rand_room_index = randrange(size)
		rand_room_name = keys[rand_room_index]
		return self.rooms[rand_room_name]["Room"]

	# Returns list of all Room objects
	def get_all_rooms(self):
		rooms = []
		for roomName in self.rooms:
			room = self.rooms[roomName]["Room"]
			rooms.append(room)

		return rooms

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
			if (self.rooms[neighbor.get_name()]["Action"] == action):
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

	def clear_room_action(self, room):
		self.rooms[room.get_name()]["Action"] = None
		room.set_action(None)

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
		self.all_combinations = []
		self.failures = 0
		self.already_colored = []

	def get_colored(self):
		return self.already_colored

	def add_colored(self, room):
		self.already_colored.append(room)

	def add_new_combination(self, new_dict):
		if len(self.all_combinations) == 0:
			self.all_combinations.append(new_dict)
			return

		unique = True

		for d in self.all_combinations:
			matching = 0
			old_keys = d.keys()

			for key in old_keys:
				if (d.get(key) == new_dict.get(key)):
					matching += 1

			if matching == len(old_keys):
				unique = False

		if unique:
			# print("\nSuccessful Mapping")
			# for key in new_dict:
			# 	try:
			# 		print("Room " + key + ": " + new_dict.get(key))
			# 	except:
			# 		print("Room " + key + ": None")
			
			self.all_combinations.append(new_dict)

	def create_dictionary(self):
		d = {}

		for room in self.floor.get_all_rooms():
			d[room.get_name()] = room.get_action()

		return d

	def get_room_action(self, room):
		actions = ["Temp", "Humidity", "Pass"]
		roomName = room.get_name()

		if roomName not in self.actionHistory:
			self.actionHistory[roomName] = []

		history = self.actionHistory[roomName]

		action_index = randrange(0,3)

		# Get first action not used

		if "Temp" in history and "Humidity" in history and "Pass" in history:
			return None

		while actions[action_index] in history:
			action_index = randrange(0,3)

		# add action to heat miser's room history
		history.append(actions[action_index])
		self.actionHistory[roomName] = history

		return actions[action_index]

	# Clears current actions done on a room
	def clear_room_history(self, room):
		self.actionHistory[room.get_name()] = []
		self.floor.clear_room_action(room)

	def get_all_combos(self):
		rooms = self.floor.get_all_rooms()
		all_combos = 0

		for room in rooms:
			# print("\n*** \nStarting in room: " + room.get_name())
			success, combos = self.preliminary_check(room)

			all_combos += combos

			# Resets floor's history
			for room in rooms:
				self.clear_room_history(room)

		# print("Total combos: ", all_combos)
        #
		# print("Total combinations: ", len(self.all_combinations))

	def reset_failures(self):
		self.failures = 0

	def add_failures(self, fails):
		self.failures += fails

	def get_failures(self):
		return self.failures

	def preliminary_check(self, startRoom):
		if startRoom is None:
			currRoom = self.floor.get_initial_room_random()
		else:
			currRoom = startRoom

		roomStack = []
		noSolution = False
		fails = 0
		successes = 0

		while not self.floor.check_floor_actions():
			action = self.get_room_action(currRoom)  # adds action to room history

			# no actions available, backtrack
			if action is None:
				fails += 1
				self.clear_room_history(currRoom)
				try:
					currRoom = roomStack.pop()
				except:
					noSolution = True
					break
			else:
				# Check whether action can be done on room
				success = self.floor.attempt_room_action(currRoom, action)

				if success:
					self.floor.set_room_action(currRoom, action)
					roomStack.append(currRoom)

					# Add all neighbors to the stack
					neighbors = currRoom.get_neighbors()
					for neighbor in neighbors:
						if neighbor.get_action() == None:
							roomStack.append(neighbor)
					currRoom = roomStack.pop() # get next room adjacent to current with no action

				# Test if pass - if so pop room to artificially
				if self.floor.check_floor_actions():
					successes += 1
					new_dict = self.create_dictionary()
					self.add_new_combination(new_dict)

					# Reset room color to force it try a new combination
					currRoom = roomStack.pop()
					self.floor.clear_room_action(currRoom)

				# No valid neighbors - backtrack
				if (currRoom is None):
					fails += 1
					currRoom = roomStack.pop()  # retrieve last room

		new_dict = self.create_dictionary()

		if not noSolution or not(None in new_dict.values()):
			self.add_new_combination(new_dict)
			return True, successes

		return False, successes

	def add_to_temperatures(self, room, changed):
		if room.get_action() == "Temp" and not(room.get_name() in changed):
			changed.append(room.get_name())

		return changed

	def add_to_humidities(self, room, changed):
		if room.get_action() == "Humidity" and not(room.get_name() in changed):
			changed.append(room.get_name())

		return changed

	def add_to_passes(self, room, changed):
		if room.get_action() == "Pass" and not(room.get_name() in changed):
			changed.append(room.get_name())

		return changed

	# Conducts brute force on all the rooms
	def brute_force_all_rooms(self):
		self.reset_failures()
		rooms = self.floor.get_all_rooms()
		changed_temps = []
		changed_humidities = []
		passes = []
		iterations = 0

		while len(changed_temps) < 10 and len(changed_humidities) < 10 and len(passes) < 10:
			success = self.brute_force(None)

			for room in rooms:
				changed_temps = self.add_to_temperatures(room, changed_temps)
				changed_humidities = self.add_to_humidities(room, changed_humidities)
				passes = self.add_to_passes(room, passes)

			# Resets floor's history
			for room in rooms:
				self.clear_room_history(room)

			iterations += 1

		print("Total brute force iterations: ", iterations)
		return iterations


	# Conducts a brute force coloring of the rooms
	# Returns if coloring was successful
	def brute_force(self, startRoom):
		if startRoom is None:
			currRoom = self.floor.get_initial_room_random()
		else:
			currRoom = startRoom

		roomStack = []
		noSolution = False
		fails = 0

		while not self.floor.check_floor_actions():
			action = self.get_room_action(currRoom) # adds action to room history

			# no actions available, backtrack
			if action is None:
				fails += 1
				self.clear_room_history(currRoom)
				try:
					currRoom = roomStack.pop()
				except:
					# fails += 1
					print("\n~~~ No solution was found! ~~~ \n")
					noSolution = True
					break
			else:
				# Check whether action can be done on room
				success = self.floor.attempt_room_action(currRoom, action)
				
				# Action valid - set action and add to stack
				if success:
					self.floor.set_room_action(currRoom, action)
					roomStack.append(currRoom)

					neighbors = currRoom.get_neighbors()
					for neighbor in neighbors:
						if neighbor.get_action() == None:
							roomStack.append(neighbor)
					currRoom = roomStack.pop() # get next room adjacent to current with no action

				# No valid neighbors - backtrack
				if (currRoom is None):
					fails += 1
					currRoom = roomStack.pop() # retrieve last room

		self.add_failures(fails)
		new_dict = self.create_dictionary()

		if not noSolution and not(None in new_dict.values()):
			return True

		return False

	def get_all_action_history(self):
		history = {}

		for room in self.floor.get_all_rooms():
			history[room.get_name()] = self.floor.rooms[room.get_name()]["Action"]

		return history

	def change_all_constrained(self):
		self.reset_failures()
		rooms = self.floor.get_all_rooms()
		previous_room_history = None
		changed_temps = []
		changed_humidities = []
		passes = []
		iterations = 0

		while len(changed_temps) < 10 and len(changed_humidities) < 10:
			previous = None
			self.floor.create_edges_stack()
			self.floor.reset_colored()

			if not(previous_room_history is None):
				first = self.floor.most_edges.pop()
				previous = previous_room_history[first]
				self.floor.most_edges.append(first)

			self.most_constraining(previous)

			for room in rooms:
				changed_temps = self.add_to_temperatures(room, changed_temps)
				changed_humidities = self.add_to_humidities(room, changed_humidities)
				passes = self.add_to_passes(room, passes)

			previous_room_history = self.get_all_action_history()
			# Resets floor's history
			for room in rooms:
				self.clear_room_history(room)

			iterations += 1

		print("Total optimized iterations: ", iterations)
		return iterations
  
	def most_constraining(self, previous):
		currRoom = self.floor.pop_edges()
		currRoomObject = self.floor.rooms[currRoom]["Room"]

		if not(previous is None):
			self.actionHistory[currRoom] = []
			self.actionHistory[currRoom].append(previous)
		noSolution = False
		fails = 0

		while not self.floor.check_floor_actions():
			action = self.get_room_action(currRoomObject)  # adds action to room history

			# no actions available, backtrack
			if action is None:
				fails += 1
				self.clear_room_history(currRoomObject)
				self.floor.most_edges.append(currRoom)

				try:
					currRoom = self.floor.already_colored.pop()
					currRoomObject = self.floor.rooms[currRoom]["Room"]

				except:
					print("~~~ No solution was found! ~~~ \n")
					noSolution = True
					break
			else:
				# Check whether action can be done on room
				success = self.floor.attempt_room_action(currRoomObject, action)

				# Action valid - set action and add to stack
				if success:
					self.floor.set_room_action(currRoomObject, action)
					self.floor.already_colored.append(currRoom)

					if len(self.floor.most_edges) != 0:
						currRoom = self.floor.most_edges.pop()

						if currRoom != None:
							# currRoom = currRoom.get_name()
							currRoomObject = self.floor.rooms[currRoom]["Room"]

				# No valid neighbors - backtrack
				if (currRoom is None):
					fails += 1
					currRoom = self.floor.already_colored.pop()  # retrieve last room
					currRoomObject = self.floor.rooms[currRoom]["Room"]

		self.add_failures(fails)
		new_dict = self.create_dictionary()

		if not noSolution and not("None" in new_dict.values()):
			return True

		return False

def main():
	heatMiser = ConstraintOptimalHeatMiser()
	print("------- FINDING ALL COMBOS -------")
	for i in range(2):
		heatMiser.get_all_combos()

	print("Total possible combinations: ", pow(3, 10))
	print("Total valid combinations found:", len(heatMiser.all_combinations))
	print("")

	for room in heatMiser.floor.get_all_rooms():
		heatMiser.floor.clear_room_action(room)

	brute_iterations = 0
	brute_failures = 0
	print("------- STARTING BRUTE FORCE -------")
	for i in range(100):
		print("BRUTE FORCE ROUND ", i+1)
		iter = heatMiser.brute_force_all_rooms()
		print("Total failures for brute force: ", heatMiser.get_failures())
		brute_iterations += iter
		brute_failures += heatMiser.failures
		print("")

	print("------- FINISHED BRUTE FORCE -------")
	print("")

	for room in heatMiser.floor.get_all_rooms():
		heatMiser.floor.clear_room_action(room)

	optimized_iterations = 0
	optimized_failures = 0
	print("------- STARTING OPTIMIZED -------")
	for i in range(100):
		print("OPTIMIZED ROUND ", i+1)
		heatMiser.floor.create_edges_stack()
		iter = heatMiser.change_all_constrained()
		print("Total failures for optimized: ", heatMiser.get_failures())
		optimized_iterations += iter
		optimized_failures += heatMiser.failures
		print("")

	print("------- FINISHED OPTIMIZED -------")
	print("")
	print("COMPARISON")
	print("Average brute force iterations: ", brute_iterations / 100)
	print("Average brute force failures: ", brute_failures / 100)
	print("")
	print("Average optimized iterations: ", optimized_iterations/ 100)
	print("Average optimized failures: ",optimized_failures/ 100)
	print("")

if __name__ == '__main__':
	main()
