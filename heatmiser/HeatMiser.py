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