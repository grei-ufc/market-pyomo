[demanda, preço, probabilidade], dayahead
def creating_scenario_file(scenario, dayaheadprice, name):
stages = len(scenario[0])
#stages = 96;

file = open('%s.dat' % name,'w') 

file.write('set I :=\n')
file.writelines('%s ' % str(i+1)  for i in range(stages))
file.write(';\n\n')

file.write('param PRICEPOOL :=\n')
for i in range(stages):
	file.write('%s %s ' % (str(i+1), str(scenario[1][i])))
file.write(';\n\n')

file.write('param DEMAND :=\n')
for i in range(stages):
	file.write('%s %s ' % (str(i+1), str(scenario[0][i])))
file.write(';\n\n')

file.write('param PRICEDAYAHEAD := %s\n' % str(dayaheadprice))
file.write(';\n')

file.close()