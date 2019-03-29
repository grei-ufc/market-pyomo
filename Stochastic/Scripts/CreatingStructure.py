scenarios = 9;
stages = 7*96;

file = open('Structure.dat','w') 

file.write('set Stages :=\n')
file.write('0')
#substituir range(stages) por range(len(scenarios[0]))
file.writelines(' %s' % str(i+1)  for i in range(stages))
file.write('\n;\n\n')

file.write('set Nodes :=\n')
file.write('n0')
for i in range(scenarios):
	for j in range(stages):
		file.write(' S%s_n%s' % (str(i+1), str(j+1)))
file.write('\n;\n\n')

file.write('param NodeStage :=\n')
file.write('n0 0')
for i in range(scenarios):
	for j in range(stages):
		file.write(' S%s_n%s %s' % (str(i+1), str(j+1), str(j+1)))
file.write('\n;\n\n')

file.write('Children[n0] :=\n')
for i in range(scenarios):
	file.write('S%s_n%s ' % (str(i+1), str(j+1)))
file.write('\n;\n')
for i in range(scenarios):
	for j in range(stages-1):
		file.write('Children[S%s_n%s] := S%s_n%s;\n' % (str(i+1), str(j+1), str(i+1), str(j+2)))
file.write('\n')

file.write('set Scenarios :=\n')
file.writelines('S%s ' % str(i+1)  for i in range(scenarios))
file.write('\n;\n\n')

file.write('param ScenarioLeafNode :=\n')
for i in range(scenarios):
		file.writelines('S%s S%s_n%s ' % (str(i+1), str(i+1), str(stages)))
file.write('\n;\n\n')

file.write('set StageVariables[0] := CONTRACTDAYAHEAD;\n')
for i in range(stages):
	file.write('set StageVariables[%s] := CONTRACTPOOL[%s];\n' % (str(i+1), str(i+1)))
file.write('\n')

file.write('param StageCost :=\n')
file.writelines('%s StageCost ' % str(i)  for i in range(stages+1))
file.write('\n;\n\n')

file.write('param ConditionalProbability :=\n')
file.write('n0 1')
for i in range(scenarios):
	for j in range(stages-1):
		file.write(' S%s_n%s 1' % (str(i+1), str(j+1)))
file.write('\n')
for i in range(scenarios):
	file.write('S%s_n1 1 ' % str(i+1))
file.write('\n;\n')

file.close()
