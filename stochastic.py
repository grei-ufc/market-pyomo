import datetime as dt
import random
import numpy as np
#import enlopy as el
import matplotlib.pyplot as plt
import pyomo.pysp.util.rapper as rapper
import pyomo.environ as pyo
import glb as glb


#Scenario generation
def generate_scenario(price_history):
	scenario = list()
	for key,value in price_history.items():
		scenario.append(random.choice(value))
	return scenario

#Scenario reduction
def reduce_scenario(scenarios, probabilities, numberofscenarios):

	selectedscenario = list()
	selectedprobabilities = list()
	total_probabilities = sum(probabilities)

	#Calcula a matriz de Kantorovich
	kantorovich = np.zeros((len(scenarios),len(scenarios)))
	for i in range(len(scenarios)):
		for j in range((len(scenarios))):
			kantorovich[i,j] = np.linalg.norm(np.asarray(scenarios[i]) - np.asarray(scenarios[j]))

	for i in range(numberofscenarios):
		#Calcula a distância de kantorovich
		kdistance = np.dot(probabilities,kantorovich)
		#Seleciona o cenário com maior semelhança
		selected = np.where(kdistance == np.min(kdistance[np.nonzero(kdistance)]))[0][0]
		selectedscenario.append(selected)
		selectedprobabilities.append(probabilities[selected])
		#Atualiza a matriz de kantorovich
		for j in range(len(scenarios)):
			if j!=selected:
				kantorovich[j,:][kantorovich[j,:]>kantorovich[j,selected]]=kantorovich[j,selected]
		probabilities[selected] = 0
		kantorovich[:,selected] = 0

	#Recalcula as probabilidades originais
	kantorovich = np.zeros((numberofscenarios,len(scenarios)))
	il = 0
	for i in selectedscenario:
		for j in range((len(scenarios))):
			kantorovich[il,j] = np.linalg.norm(np.asarray(scenarios[i]) - np.asarray(scenarios[j]))
		il += 1

	decision = np.argmin(kantorovich, axis=0)
	for i in range(len(decision)):
		#print('testando',i)
		if i not in selectedscenario:
			#print('esta em',i)
			#print('adicionando a probabildade de',i,'em',selectedscenario[decision[i]])
			#print(selectedprobabilities[decision[i]],'+',probabilities[i])			
			selectedprobabilities[decision[i]] = selectedprobabilities[decision[i]]+probabilities[i]


	#Normalize probabilities
	selectedscenarios = list()
	for i in range(len(selectedprobabilities)):
		selectedprobabilities[i] = selectedprobabilities[i]/total_probabilities
		selectedscenarios.append(scenarios[selectedscenario[i]])

	#print('sum of final probabilities ', sum(selectedprobabilities))
	return selectedscenarios, selectedprobabilities


def combine_scenarios(scenario_tuple1, scenario_tuple2):
	all_scenarios = list()
	for j in range(len(scenario_tuple2[0])):
		for i in range(len(scenario_tuple1[0])):			
			#print('adicionando demanda', i+1, 'preço', j+1, 'probabilidade', (i+1),'*',(j+1))
			single_scenario = [scenario_tuple1[0][i], scenario_tuple2[0][j], scenario_tuple1[1][i]*scenario_tuple2[1][j]]
			all_scenarios.append(single_scenario)
			#print(len(all_scenarios))

	return all_scenarios


def change_data_files(all_scenarios, dayaheadprice, path):
	probabilities = list()
	for i in range(len(all_scenarios)):
		probabilities.append(all_scenarios[i][2])
		name = 'S%s' % str(i+1)
		_create_scenario_data(all_scenarios[i], dayaheadprice, name, path)
	_change_scenario_structure(probabilities, path)


def optimize_contract(path):	
	stsolver = rapper.StochSolver(path+'/ReferenceModel.py', fsfct = None, tree_model = path+'/ScenarioStructure.dat', phopts = None)
	ef_sol = stsolver.solve_ef('cplex')
	'''
	while ef_sol.solver.termination_condition !=  pyo.TerminationCondition.optimal:
		print('solving')
		print(ef_sol.solver.termination_condition)
	print('solved')
	'''
	return stsolver


def get_optimal_value(stsolver):
	opt_value = dict()
	for varname, varval in stsolver.root_Var_solution():
		opt_value[varname] = varval
	return opt_value 


def _change_scenario_structure(probabilities, path):
	MYFILE = path+'/ScenarioStructure.dat'

	lines = open(MYFILE, 'r').readlines()

	new_addition = str()
	for i in range(len(probabilities)):
		new_addition = new_addition + ('S%s_n1 %s ' % (str(i+1), str(probabilities[i])))
	new_addition = new_addition + '\n'
	lines[-2] = new_addition

	open(MYFILE, 'w').writelines(lines)


def _create_scenario_data(scenario, dayaheadprice, name, path):
	stages = len(scenario[0])

	file = open(path+'/%s.dat' % name,'w') 

	file.write('set I :=\n')
	file.writelines('%s ' % str(i+1)  for i in range(stages))
	file.write(';\n\n')

	Ta = 1
	for i in range(glb.tariffs_during_one_day):
		file.write('set T%s :=\n' %str(i+1))
		T = list()
		for day in range(glb.contract_market_interval):
			T.extend([(i+Ta+day*glb.stages_during_one_day) for i in range(int(glb.stages_during_one_day/glb.tariffs_during_one_day))])
		file.writelines(' %s' %str(j) for j in T)
		file.write('\n;\n\n')
		Ta = Ta + int(glb.stages_during_one_day/glb.tariffs_during_one_day)
		
	file.write('param PRICEPOOL :=\n')
	for i in range(stages):
		file.write('%s %s ' % (str(i+1), str(scenario[1][i])))
	file.write(';\n\n')

	file.write('param DEMAND :=\n')
	for i in range(stages):
		file.write('%s %s ' % (str(i+1), str(scenario[0][i])))
	file.write(';\n\n')

	for i in range(len(dayaheadprice)):
		file.write('param PRICEDAYAHEADT%s := %s\n' % (str(i+1), str(dayaheadprice[i])))
		file.write(';\n')

	file.close()

def create_scenario_structure(stages, scenarios, path):

	file = open(path+'/ScenarioStructure.dat','w') 

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

	file.write('set Children[n0] :=\n')
	for i in range(scenarios):
		file.write('S%s_n%s ' % (str(i+1), str(1)))
	file.write('\n;\n')
	for i in range(scenarios):
		for j in range(stages-1):
			file.write('set Children[S%s_n%s] := S%s_n%s;\n' % (str(i+1), str(j+1), str(i+1), str(j+2)))
	file.write('\n')

	file.write('set Scenarios :=\n')
	file.writelines('S%s ' % str(i+1)  for i in range(scenarios))
	file.write('\n;\n\n')

	file.write('param ScenarioLeafNode :=\n')
	for i in range(scenarios):
			file.writelines('S%s S%s_n%s ' % (str(i+1), str(i+1), str(stages)))
	file.write('\n;\n\n')

	file.write('set StageVariables[0] := CONTRACTDAYAHEADT1 CONTRACTDAYAHEADT2 CONTRACTDAYAHEADT3 CONTRACTDAYAHEADT4;\n')
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
			file.write(' S%s_n%s 1' % (str(i+1), str(j+2)))
	file.write('\n')
	for i in range(scenarios):
		file.write('S%s_n1 1 ' % str(i+1))
	file.write('\n;\n')

	file.close()

'''
if __name__ == '__main__':


#create price history

#step de simulação em minutos
simulationstep = 15

#cria o dicionário que tem o banco de dados do histórico de preço
stages = [dt.datetime.strptime('00:00:00','%H:%M:%S') + dt.timedelta(0,t*simulationstep*60) for t in range(0, 96, 1)]
for i in range(len(stages)):
	stages[i] = stages[i].time()
price_history = dict()
for i in stages:
	price_history[i] = list()

#gerando um histórico de preço 
load_curve1 = el.gen_daily_stoch_el(100)
load_curve2 = el.gen_daily_stoch_el(100)
load_curve3 = el.gen_daily_stoch_el(100)
load_curve4 = el.gen_daily_stoch_el(100)

load_curve = list()
for i in range(len(load_curve1)):
	load_curve.append(load_curve1[i]+random.uniform(0,1))
	load_curve.append(load_curve2[i]+random.uniform(0,1))
	load_curve.append(load_curve3[i]+random.uniform(0,1))
	load_curve.append(load_curve4[i]+random.uniform(0,1))

#plt.plot(load_curve)

for i in range(len(stages)):
	for j in range(2000):
		price_history[stages[i]].append(load_curve[i] + random.uniform(0,4))

#scenarios = np.genfromtxt('Dados.csv', delimiter=';')
'''

'''
for i in scenarios:
	plt.plot(i)

'''

'''
for i in selectedscenarios:
	plt.plot(scenarios[i])
'''



#importa
#gera os dados aleatórios
#plt.plot(load_curve)

'''
def function():
	#iniciar price hist
	total = list()
	for n in range(250):
		total.append(generate_scenario(price_history))

	#for i in total:
	#    plt.plot(i)

	scenarios = total

	#for i in range(len(scenarios)):
	#    plt.plot(scenarios[i])


	probabilities = np.ones(len(scenarios))
	prices = reduce_scenario(scenarios, probabilities, 3)

	#for i in prices[0]:
	#    plt.plot(i)

	scenarios = np.genfromtxt('Dados.csv', delimiter=';')
	probabilities = np.ones(len(scenarios))
	demand = reduce_scenario(scenarios, probabilities, 3)

	#for i in demand[0]:
	#    plt.plot(i)

	all_scenarios = combine_scenarios(demand, prices)

	create_scenario_structure(len(all_scenarios[0][0]), len(all_scenarios))

	change_data_files(all_scenarios, 4)

	path = '/home/talesmt/Stochastic/Join'
	optimize_contract(path)
'''