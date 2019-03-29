stages = 96;

file = open('Scenario.dat','w') 

file.write('set I :=\n')
file.writelines('%s ' % str(i+1)  for i in range(stages))
file.write(';\n\n')

file.write('param PRICEPOOL :=\n')
for i in range(stages):
	file.write('%s P%s ' % (str(i+1), str(i+1)))
file.write(';\n\n')

file.write('param DEMAND :=\n')
for i in range(stages):
	file.write('%s D%s ' % (str(i+1), str(i+1)))
file.write(';\n\n')

file.write('param PRICEDAYAHEAD := P%s\n' % str(1))
file.write(';\n')

file.close()