probabilities
scenarios = 9;
MYFILE='ScenarioStructure.dat'

# read the file into a list of lines
lines = open(MYFILE, 'r').readlines()

# now edit the last line of the list of lines
new_addition = str()
for i in range(scenarios):
	new_addition = new_addition + ('S%s_n1 %s ' % (str(i+1), str(probabilities[i])))
new_addition = new_addition + '\n'
lines[-2] = new_addition

# now write the modified list back out to the file
open(MYFILE, 'w').writelines(lines)