import datetime as dt
import csv
import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import glb as glb
from devices import *
from stochastic import *
import pandas as pd
from graph import *


class Prosumer(object):

	def __init__(self, id, *devices):

		self.type = 'Prosumer'
		self.id = str(id)

		self.devices = list()
		for device in devices:
			self.devices.append(device)

		self.utility_contracted_power = [0 for i in range(glb.tariffs_during_one_day)]
		self.utility_used_contracted_power = list()
		self.difference_curve = list()
		self.power = list()
		self.price = list()
		self.curve = list()

		self.reduced_expense = 0
		self.increased_expense = 0
		self.reduced_income = 0
		self.increased_income = 0
		self.paid_price = 0

	def get_bid(self, time):
		self.curve = list()
		for i in self.devices:
			self.curve.append(i.get_bid(time))

		tariff = len(glb.tariff_hours)-1
		for i in range(len(glb.tariff_hours) - 1):
			if time.time() >= glb.tariff_hours[i] and time.time() < glb.tariff_hours[i + 1]:
				tariff = i

		self.difference_curve = list()
		self.curve = np.sum(self.curve, axis=0)
		for i in range(len(self.curve)):
			if self.curve[i] >= 0:
				if self.curve[i] <= self.utility_contracted_power[tariff]:
					self.difference_curve.append(self.curve[i])                 
					self.curve[i] = 0
				else:
					self.difference_curve.append(self.utility_contracted_power[tariff])
					self.curve[i] = self.curve[i] - \
						self.utility_contracted_power[tariff]
			else:
				self.difference_curve.append(0)

		return self.curve

	def inform_price(self, price_position):
		self.device_power = 0
		for i in self.devices:
			power = i.set_power(price_position)
			self.device_power = self.device_power + power
		self.set_power(price_position)

	def set_power(self, price_position):
		self.power.append(self.curve[price_position])
		self.price.append(price_position / glb.money_units)
		self.utility_used_contracted_power.append(self.difference_curve[price_position])

		paid_price = self.curve[price_position] * \
			price_position / glb.money_units
		self.paid_price = self.paid_price + paid_price
		willing_to_have_paid = ((np.asarray(
			self.curve) - self.curve[0] + 0.005).argmin() / glb.money_units) * self.curve[0]
		if (paid_price < willing_to_have_paid) and (paid_price > 0):
			self.reduced_expense = self.reduced_expense + \
				(willing_to_have_paid - paid_price)
		elif(paid_price > willing_to_have_paid) and (paid_price > 0):
			self.increased_expense = self.increased_expense + \
				(- willing_to_have_paid + paid_price)
		elif (paid_price < willing_to_have_paid) and (paid_price < 0):
			self.reduced_income = self.reduced_expense + \
				(willing_to_have_paid - paid_price)
		elif (paid_price > willing_to_have_paid) and (paid_price < 0):
			self.increased_income = self.increased_expense + \
				(- willing_to_have_paid + paid_price)

	def combine_device_scenarios(self):
		self.demand_scenario = list()
		for i in self.devices:
			self.demand_scenario.append(i.reduced_scenario)

		all_demand_scenarios = list()
		all_demand_scenarios = self.demand_scenario.pop(0)

		while self.demand_scenario:
			intermediate_demand_scenario = (list(), list())
			for i in range(len(all_demand_scenarios[0])):
				for j in range(len(self.demand_scenario[0][0])):
					intermediate_demand_scenario[0].append([sum(x) for x in zip(
						all_demand_scenarios[0][i], self.demand_scenario[0][0][j])])
					intermediate_demand_scenario[1].append(
						all_demand_scenarios[1][i] * self.demand_scenario[0][1][j])
			all_demand_scenarios = intermediate_demand_scenario[:]
			self.demand_scenario.pop(0)

		if len(all_demand_scenarios[0]) > 3:
			self.demand_scenario = reduce_scenario(
				all_demand_scenarios[0], all_demand_scenarios[1], 3)
		else:
			self.demand_scenario = all_demand_scenarios

	def combine_scenario(self, price_scenario):
		return combine_scenarios(self.demand_scenario, price_scenario)

	def solve_stochastic(self, utility_demand_price):
		print('stochastic de', self.id)
		create_scenario_structure(len(self.all_scenarios[0][0]), len(
			self.all_scenarios), glb.stochastic_path)
		change_data_files(self.all_scenarios,
						  utility_demand_price, glb.stochastic_path)
		stsolver = optimize_contract(glb.stochastic_path)
		self.set_utility_contracted_power(get_optimal_value(stsolver))

	def stochastic_time(self, stages, price_scenario, utility_demand_price):
		for i in self.devices:
			i.generate_scenario()
		self.combine_device_scenarios()
		self.all_scenarios = self.combine_scenario(price_scenario)
		self.solve_stochastic(utility_demand_price)

	def set_utility_contracted_power(self, contracted):
		print('ag', self.id, ' ', contracted)
		self.utility_contracted_power[0] = contracted['CONTRACTDAYAHEADT1']
		self.utility_contracted_power[1] = contracted['CONTRACTDAYAHEADT2']
		self.utility_contracted_power[2] = contracted['CONTRACTDAYAHEADT3']
		self.utility_contracted_power[3] = contracted['CONTRACTDAYAHEADT4']

	def store_data(self, data, time):
		data.append([self.type, self.id, 'Energy (kWh)', time.strftime('%d/%m/%Y')] +
					[i for i in self.power[:]] + ['Total (kWh):'] + [sum([i for i in self.power[:]])])
		for i in self.devices:
			i.store_data(data, time, self.id)
		data.append([self.type, self.id, 'Short Market Price ($MU)', time.strftime('%d/%m/%Y')] + [a * b for a, b in zip(
			self.power[:], self.price[:])] + ['Total ($MU):'] + [sum([a * b for a, b in zip(self.power[:], self.price[:])])])
		data.append([self.type, self.id, 'Contracted Energy (kWh)', time.strftime(
			'%d/%m/%Y')] + [val for val in self.utility_contracted_power[:] for _ in range(24)])
		data.append([self.type, self.id, 'Long Market Price ($MU) - Corrigir', time.strftime('%d/%m/%Y')] + [a * b for a, b in zip(self.utility_contracted_power[:], [glb.contract_tariff_1 * 24, glb.contract_tariff_2 * 24, glb.contract_tariff_3 * 24, glb.contract_tariff_4 * 24])
																											 ] + [0 for _ in range(96 - 4)] + ['Total ($MU):'] + [sum([a * b for a, b in zip(self.utility_contracted_power[:], [glb.base_price_tariff_1 * 24, glb.base_price_tariff_2 * 24, glb.base_price_tariff_3 * 24, glb.base_price_tariff_4 * 24])])])
		data.append([self.type, self.id,'Energy used from Long Market (kWh)', time.strftime('%d/%m/%Y')] + [val for val in self.utility_used_contracted_power[:]])

		self.power = list()
		self.price = list()
		self.utility_used_contracted_power = list()

	def data_setting(self, number):
		for i in self.devices:
			i.data_setting(number)

	def store_total_data(self, total_data):
		total_data.append(
			[self.type, self.id, 'Reduced Expense ($MU)', self.reduced_expense])
		total_data.append(
			[self.type, self.id, 'Increased Expense ($MU)', self.increased_expense])
		total_data.append(
			[self.type, self.id, 'Reduced Income ($MU)', self.reduced_income])
		total_data.append(
			[self.type, self.id, 'Increased Income ($MU)', self.increased_income])
		total_data.append(
			[self.type, self.id, 'Total Paid Price ($MU)', self.paid_price])
		for i in self.devices:
			i.store_total_data(total_data, self.id)

	def store_instantiation_data(self, instantiation_data):
		has_pv_generation = False
		for i in self.devices:
			if i.type == 'PVGeneration':
				has_pv_generation = True
			i.store_instantiation_data(instantiation_data, self.id)

		if has_pv_generation:
			instantiation_data.append('ag%s = Prosumer(%s,pv%s,ul%s)' %(str(self.id),str(self.id),str(self.id),str(self.id)))
		else:
			instantiation_data.append('ag%s = Prosumer(%s,ul%s)' %(str(self.id),str(self.id),str(self.id)))


class Utility(object):

	def __init__(self):
		self.type = 'Utility'
		self.id = None

		self.utility_demand_price = [glb.contract_market_interval * i *
									 glb.stages_during_one_day / glb.tariffs_during_one_day for i in glb.contract_tariffs]
		self.power = list()
		self.price = list()
		self.curves = list()
		for curve_mask in glb.ut_bid_curves:
			self.curves.append([i for i in curve_mask])

		'''
		self.utility_demand_price = [glb.contract_tariff_1*24*glb.contract_market_interval, \
										glb.contract_tariff_2*24*glb.contract_market_interval, \
										glb.contract_tariff_3*24*glb.contract_market_interval, \
										glb.contract_tariff_4*24*glb.contract_market_interval]

		'''

		self.tariff_hours = list()
		for hour in glb.tariff_hours:
			self.tariff_hours.append(hour)

		self.paid_price = 0

		self.line = list()
		self.x_vec = np.linspace(0, 1, 100)[0:-1]
		self.vol_vec = [0] * len(self.x_vec)

	def get_device_daily_power(self):
		pass

	def get_bid(self, time):

		tariff = len(self.tariff_hours) - 1
		for tarifftime in range(len(self.tariff_hours) - 1):
			# print(time.time()  >= self.tariff_hours[tarifftime])
			# print(time.time()  < self.tariff_hours[tarifftime +1])
			if time.time() >= self.tariff_hours[tarifftime] and time.time() < self.tariff_hours[tarifftime + 1]:
				tariff = tarifftime
		# print(dt.datetime.strftime(time,'%H:%M:%S'))
		# print("utility ofertando tarifa ", tariff+1)
		self.curve = [i for i in self.curves[tariff]]
		return self.curve

	def stochastic_time(self, stages, price_scenario, utility_demand_price):
		pass

	def data_setting(self, number):
		pass

	def inform_price(self, price_position):
		#self.price = price_position/glb.money_units
		#print("preço do mercado spot:", price_position / 100)
		#print("preço do mercado semanal:", glb.contract_tariff_1,
		#      glb.contract_tariff_2, glb.contract_tariff_3, glb.contract_tariff_4)
		self.power.append(self.curve[price_position])
		self.price.append(price_position / glb.money_units)
		self.update_graph()

		paid_price = self.curve[price_position] * \
			price_position / glb.money_units
		self.paid_price = self.paid_price + paid_price

	def store_data(self, data, time):
		data.append([self.type, self.id, 'Energy (kWh)', time.strftime('%d/%m/%Y')] +
					[i for i in self.power[:]] + ['Total (kWh):'] + [sum([i for i in self.power[:]])])
		data.append([self.type, self.id, 'Short Market Price ($MU)', time.strftime('%d/%m/%Y')] + [a * b for a, b in zip(
			self.power[:], self.price[:])] + ['Total ($MU):'] + [sum([a * b for a, b in zip(self.power[:], self.price[:])])])
		self.power = list()
		self.price = list()

	def update_graph(self):
		return
		self.vol_vec = np.append(self.vol_vec[1:], self.power[-1])
		self.line = live_plotter(
			self.x_vec, self.vol_vec, self.line, plot_position=122, ylabel='Energy')

	def store_total_data(self, total_data):
		total_data.append(
			[self.type, self.id, 'Total Paid Price ($MU)', self.paid_price])

	def store_instantiation_data(self, instantiation_data):
		instantiation_data.append('ut = Utility()')


class Aggregator(object):

	def __init__(self):
		self.type = 'Aggregator'
		self.id = None
		self.bids = list()
		self.aggregated_curve = list()
		self.opt_price_position = None
		self.price = None
		self.price_difference = None
		self.data_price = list()
		self.temporary_daily_price = list()

	def ask_for_bids(self, time, agents):
		self.bids = list()
		for i in agents:
			self.bids.append(i.get_bid(time))

	def sum_bids(self):
		self.aggregated_curve = np.sum(self.bids, axis=0)
		# plt.plot(glb.price_list, self.aggregated_curve)

	def find_optimal_price(self, agents):
		i = np.abs(self.aggregated_curve - 0).argmin()
		self.opt_price_position = i
		self.price = i / glb.money_units
		self.temporary_daily_price.append(self.price)
		self.price_difference = self.aggregated_curve[i]
		for agent in agents:
			agent.inform_price(self.opt_price_position)

	def operate_stage(self, time, agents):
		self.ask_for_bids(time, agents)
		self.sum_bids()
		self.find_optimal_price(agents)

	def store_data(self, data, time):
		data.append([self.type, self.id, 'Short Market Clearing Price ($MU)', time.strftime(
			'%d/%m/%Y')] + [i for i in self.temporary_daily_price[:]])
		self.data_price.append(self.temporary_daily_price[:])
		self.temporary_daily_price = list()

	def generate_scenarios(self):
		scenarios = list()
		x = np.linspace(0, 3.5, len(self.data_price))
		p = stats.halfnorm.pdf(x)[::-1] / stats.halfnorm.pdf(x).sum()
		# plt.plot(p)
		for i in range(20):
			create_scenario = list()
			for i in range(glb.contract_market_interval):
				create_scenario.extend(
					self.data_price[np.random.choice(len(self.data_price), p=p)])
			scenarios.append(create_scenario[:])
		probabilities = np.ones(len(scenarios))
		try:
			self.price_scenario = reduce_scenario(scenarios, probabilities, 3)
			return self.price_scenario
		except:
			selected_scenario = list()
			selected_scenario.append(
				glb.contract_market_interval * random.choice(self.data_price))
			selected_scenario.append(
				glb.contract_market_interval * random.choice(self.data_price))
			selected_scenario.append(
				glb.contract_market_interval * random.choice(self.data_price))
			probabilities = [1 / 3, 1 / 3, 1 / 3]
			return selected_scenario, probabilities

	def store_instantiation_data(self, instantiation_data):
		instantiation_data.append('agr = Aggregator()')


class Controller(object):

	def __init__(self, start_time, finish_time, stochastic_time, step, aggregator, agents):
		self.time = dt.datetime.strptime(start_time, '%d/%m/%Y - %H:%M:%S')
		self.agents = list()
		for agent in agents:
			self.agents.append(agent)
		self.finish_time = dt.datetime.strptime(
			finish_time, '%d/%m/%Y - %H:%M:%S')
		self.step = dt.timedelta(minutes=step)
		self.aggregator = aggregator
		self.stochastic_time = dt.datetime.strptime(
			stochastic_time, '%d/%m/%Y - %H:%M:%S')
		self.stages = 24 * 60 / step
		self.data = list()
		self.total_data = list()
		self.instantiation_data = list()

	def pool_time(self):
		self.aggregator.operate_stage(self.time, self.agents)

	def call_stochastic(self):
		for i in self.agents:
			if i.type == 'Utility':
				utility_demand_price = i.utility_demand_price
		for i in self.agents:
			print(dt.datetime.strftime(self.time, '%d/%m/%Y'))
			price_scenario = self.aggregator.generate_scenarios()
			i.stochastic_time(self.stages, price_scenario,
							  utility_demand_price)

	def data_setting(self):
		number = np.random.randint(0, len(glb.data_pv_generation) - 1)
		for i in self.agents:
			i.data_setting(number)

	def store_data(self):
		self.aggregator.store_data(self.data, self.time)
		for i in self.agents:
			i.store_data(self.data, self.time)

	def store_total_data(self):
		for i in self.agents:
			i.store_total_data(self.total_data)

	def store_instantiation_data(self):
		for i in self.agents:
			i.store_instantiation_data(self.instantiation_data)
		self.aggregator.store_instantiation_data(self.instantiation_data)

	def load_data(self):

		for i in range(15):
			path = "/home/talesmt/Stochastic/WorkSpace/Data/UserLoad/UL%s.csv" % str(
				i + 1)
			temp = pd.read_csv(path, index_col=False,
							   header=None, delimiter=';')
			k = temp.values.tolist()
			# print(type(k[0][0]))
			# print('k', k[:])
			# print(k)
			# print(temp.values.tolist())
			glb.data_user_load.append(k[:])
			# print(type(glb.data_user_load[i][0][0]))

		for i in range(3):
			temp = list()
			path = "/home/talesmt/Stochastic/WorkSpace/Data/PVGeneration/PV%s.csv" % str(
				i + 1)
			temp = pd.read_csv(path, index_col=False,
							   header=None, delimiter=';')
			glb.data_pv_generation.append(temp.values.tolist())

		path = "/home/talesmt/Stochastic/WorkSpace/Data/Storage/Storage_t.csv"
		with open(path, "r") as f:
			reader = csv.reader(f, delimiter=";", quoting=csv.QUOTE_NONNUMERIC)
			# glb.data_storage
			glb.data_storage = list(reader)

		# global userload_mapping
		glb.userload_mapping = {'23:45': 95, '23:30': 94, '23:15': 93, '23:00': 92, '22:45': 91, '22:30': 90, '22:15': 89, '22:00': 88, '21:45': 87, '21:30': 86, '21:15': 85, '21:00': 84, '20:45': 83, '20:30': 82, '20:15': 81, '20:00': 80, '19:45': 79, '19:30': 78, '19:15': 77, '19:00': 76, '18:45': 75, '18:30': 74, '18:15': 73, '18:00': 72, '17:45': 71, '17:30': 70, '17:15': 69, '17:00': 68, '16:45': 67, '16:30': 66, '16:15': 65, '16:00': 64, '15:45': 63, '15:30': 62, '15:15': 61, '15:00': 60, '14:45': 59, '14:30': 58, '14:15': 57, '14:00': 56, '13:45': 55, '13:30': 54, '13:15': 53, '13:00': 52, '12:45': 51, '12:30': 50, '12:15': 49,
								'12:00': 48, '11:45': 47, '11:30': 46, '11:15': 45, '11:00': 44, '10:45': 43, '10:30': 42, '10:15': 41, '10:00': 40, '09:45': 39, '09:30': 38, '09:15': 37, '09:00': 36, '08:45': 35, '08:30': 34, '08:15': 33, '08:00': 32, '07:45': 31, '07:30': 30, '07:15': 29, '07:00': 28, '06:45': 27, '06:30': 26, '06:15': 25, '06:00': 24, '05:45': 23, '05:30': 22, '05:15': 21, '05:00': 20, '04:45': 19, '04:30': 18, '04:15': 17, '04:00': 16, '03:45': 15, '03:30': 14, '03:15': 13, '03:00': 12, '02:45': 11, '02:30': 10, '02:15': 9, '02:00': 8, '01:45': 7, '01:30': 6, '01:15': 5, '01:00': 4, '00:45': 3, '00:30': 2, '00:15': 1, '00:00': 0}
		# global pvgeneration_mapping
		glb.pvgeneration_mapping = {'23:45': 95, '23:30': 94, '23:15': 93, '23:00': 92, '22:45': 91, '22:30': 90, '22:15': 89, '22:00': 88, '21:45': 87, '21:30': 86, '21:15': 85, '21:00': 84, '20:45': 83, '20:30': 82, '20:15': 81, '20:00': 80, '19:45': 79, '19:30': 78, '19:15': 77, '19:00': 76, '18:45': 75, '18:30': 74, '18:15': 73, '18:00': 72, '17:45': 71, '17:30': 70, '17:15': 69, '17:00': 68, '16:45': 67, '16:30': 66, '16:15': 65, '16:00': 64, '15:45': 63, '15:30': 62, '15:15': 61, '15:00': 60, '14:45': 59, '14:30': 58, '14:15': 57, '14:00': 56, '13:45': 55, '13:30': 54, '13:15': 53, '13:00': 52, '12:45': 51, '12:30': 50, '12:15': 49,
									'12:00': 48, '11:45': 47, '11:30': 46, '11:15': 45, '11:00': 44, '10:45': 43, '10:30': 42, '10:15': 41, '10:00': 40, '09:45': 39, '09:30': 38, '09:15': 37, '09:00': 36, '08:45': 35, '08:30': 34, '08:15': 33, '08:00': 32, '07:45': 31, '07:30': 30, '07:15': 29, '07:00': 28, '06:45': 27, '06:30': 26, '06:15': 25, '06:00': 24, '05:45': 23, '05:30': 22, '05:15': 21, '05:00': 20, '04:45': 19, '04:30': 18, '04:15': 17, '04:00': 16, '03:45': 15, '03:30': 14, '03:15': 13, '03:00': 12, '02:45': 11, '02:30': 10, '02:15': 9, '02:00': 8, '01:45': 7, '01:30': 6, '01:15': 5, '01:00': 4, '00:45': 3, '00:30': 2, '00:15': 1, '00:00': 0}
		# global storage_mapping
		glb.pvgeneration_mapping = {'23:45': 95, '23:30': 94, '23:15': 93, '23:00': 92, '22:45': 91, '22:30': 90, '22:15': 89, '22:00': 88, '21:45': 87, '21:30': 86, '21:15': 85, '21:00': 84, '20:45': 83, '20:30': 82, '20:15': 81, '20:00': 80, '19:45': 79, '19:30': 78, '19:15': 77, '19:00': 76, '18:45': 75, '18:30': 74, '18:15': 73, '18:00': 72, '17:45': 71, '17:30': 70, '17:15': 69, '17:00': 68, '16:45': 67, '16:30': 66, '16:15': 65, '16:00': 64, '15:45': 63, '15:30': 62, '15:15': 61, '15:00': 60, '14:45': 59, '14:30': 58, '14:15': 57, '14:00': 56, '13:45': 55, '13:30': 54, '13:15': 53, '13:00': 52, '12:45': 51, '12:30': 50, '12:15': 49,
									'12:00': 48, '11:45': 47, '11:30': 46, '11:15': 45, '11:00': 44, '10:45': 43, '10:30': 42, '10:15': 41, '10:00': 40, '09:45': 39, '09:30': 38, '09:15': 37, '09:00': 36, '08:45': 35, '08:30': 34, '08:15': 33, '08:00': 32, '07:45': 31, '07:30': 30, '07:15': 29, '07:00': 28, '06:45': 27, '06:30': 26, '06:15': 25, '06:00': 24, '05:45': 23, '05:30': 22, '05:15': 21, '05:00': 20, '04:45': 19, '04:30': 18, '04:15': 17, '04:00': 16, '03:45': 15, '03:30': 14, '03:15': 13, '03:00': 12, '02:45': 11, '02:30': 10, '02:15': 9, '02:00': 8, '01:45': 7, '01:30': 6, '01:15': 5, '01:00': 4, '00:45': 3, '00:30': 2, '00:15': 1, '00:00': 0}

	def increase_time(self):
		self.time = self.time + self.step

	def create_output_data_file(self):
		path = "/home/talesmt/Stochastic/WorkSpace/Data/Output/Output.csv"
		with open(path, "w") as f:
			writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL)
			for line in self.data:
				writer.writerow(line)

		self.store_total_data()
		path = "/home/talesmt/Stochastic/WorkSpace/Data/Output/Output2.csv"
		with open(path, "w") as f:
			writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_ALL)
			for line in self.total_data:
				writer.writerow(line)

		self.store_instantiation_data()
		path = "/home/talesmt/Stochastic/WorkSpace/Data/Output/Instantiation.txt"
		with open(path, 'w') as f:
			for item in self.instantiation_data:
				f.write("%s\n" % item)


	def update_graph(self):
		return
		self.price_vec = np.append(self.price_vec[1:], self.aggregator.price)
		self.line = live_plotter(self.x_vec, self.price_vec, self.line)

	def simulation(self, start_time, finish_time, step):
		#config()

		self.time = dt.datetime.strptime(start_time, '%d/%m/%Y - %H:%M:%S')
		self.finish_time = dt.datetime.strptime(
			finish_time, '%d/%m/%Y - %H:%M:%S')
		self.step = dt.timedelta(minutes=step)
		self.stages = 24 * 60 / step
		flag_end = 0

		self.store_time = dt.datetime.strptime(
			"01/01/2000 - 23:45:00", '%d/%m/%Y - %H:%M:%S').time()
		stochastic_hour = '23:45:00'
		start = self.time.strftime('%d/%m/%Y') + ' - ' + stochastic_hour
		finish = self.finish_time.strftime(
			'%d/%m/%Y') + ' - ' + stochastic_hour
		self.stochastic_time = glb.generate_timeseries(
			start_time, finish_time, glb.contract_market_interval)
		self.stochastic_time.pop(0)

		self.line = list()
		self.x_vec = np.linspace(0, 1, 100)[0:-1]
		self.price_vec = [0] * len(self.x_vec)

		self.load_data()
		self.data_setting()
		print(dt.datetime.strftime(self.time, '%d/%m/%Y'))
		while flag_end == 0:
			self.pool_time()
			self.update_graph()
			if self.time == self.finish_time:
				flag_end = 1
			else:
				if self.time.time() == self.store_time:
					self.store_data()
					self.data_setting()
				if self.time in self.stochastic_time:
					print(dt.datetime.strftime(self.time, '%d/%m/%Y'))
					self.call_stochastic()
			self.increase_time()
		self.create_output_data_file()


if __name__ == '__main__':

	'''

			for i in range (4):
					controller.data_setting()
					for i in range(96):
							controller.pool_time()
							controller.time = controller.time + 15 

	'''

	'''controller = Controller(agent_list, start_time, finish_time, step)
	controller.simulation()
	'''
	'''
	from agents import *

	pv1 = PVGeneration(2.0)
	ul1 = UserLoad()
	ag1 = Prosumer(1,pv1,ul1)

	pv2 = PVGeneration(2.5)
	ul2 = UserLoad()
	ag2 = Prosumer(2,pv2,ul2)

	pv3 = PVGeneration(1.5)
	ul3 = UserLoad()
	ag3 = Prosumer(3,pv3,ul3)

	pv4 = PVGeneration(3.5)
	ul4 = UserLoad()
	ag4 = Prosumer(4,pv4,ul4)

	ul5 = UserLoad()
	ag5 = Prosumer(5,ul5)

	pv6 = PVGeneration(4.0)
	ul6 = UserLoad()
	ag6 = Prosumer(6,pv6,ul6)

	pv7 = PVGeneration(1.0)
	ul7 = UserLoad()
	ag7 = Prosumer(7,pv7,ul7)

	ul8 = UserLoad()
	ag8 = Prosumer(8,ul8)

	pv9 = PVGeneration(3.5)
	ul9 = UserLoad()
	ag9 = Prosumer(9,pv9,ul9)

	pv10 = PVGeneration(2.5)
	ul10 = UserLoad()
	ag10 = Prosumer(10,pv10,ul10)

	ut = Utility()

	agr = Aggregator()

	start_time = '01/01/2019 - 00:00:00'
	finish_time = '04/01/2019 - 00:00:00'
	stochastic_time = '01/01/2019 - 23:45:00'
	step = 15
	controller = Controller(start_time, finish_time, stochastic_time, step, agr, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9, ag10, ut)

	controller.load_data()
	controller.data_setting()
	controller.pool_time()
	controller.store_data()

	for i in ag1.devices:
		i.generate_scenario()

	ag1.combine_device_scenarios()
	'''

	# def store_data(self, data, time):
	# pass
	'''
			data.append([self.type, self.id, time.strftime('%d/%m/%Y')]+[i for i in self.power[:]])
			self.power = list()
			for i in self.devices:
				print(i.type)
				# i.store_data(data, time, self.id)
	'''

	'''
			data.append([self.type, self.id, time.strftime('%d/%m/%Y')]+[i for i in self.power[:]])
			self.power = list()
			for i in self.devices:
				i.store_data(data, time, self.id)
	'''

	def create(n, start_time='01/01/2019 - 00:00:00', finish_time='15/02/2019 - 00:00:00', stochastic_time='01/01/2019 - 23:45:00', step=15):
		agents = list()
		for i in range(n):
			if uniform(0, 1) <= 0.3:
				x = uniform(1, 3.5)
				agents.append(Prosumer(i + 1, UserLoad(), PVGeneration(x)))
			else:
				agents.append(Prosumer(i + 1, UserLoad()))
		agents.append(Utility())
		agr = Aggregator()
		controller = Controller(start_time, finish_time,
								stochastic_time, step, agr, agents)
		for i in agents:
			print(i.id)
		return controller

	start_time = '01/01/2019 - 00:00:00'
	finish_time = '01/03/2019 - 00:00:00'
	stochastic_time = '01/01/2019 - 23:45:00'
	step = 15

	controller = create(200)

	controller.simulation(start_time, finish_time, step)
	'''
	for i in range(100):
		print("ul%s = UserLoad()" % str(i+1))
		if uniform(0,1) <= 0.3:
			x = uniform(1,3.5)
			print("pv%s = PVGeneration(%.2f)" % (str(i+1),x))
			print("ag%s = Prosumer(%s,pv%s,ul%s)" % (str(i+1), str(i+1), str(i+1), str(i+1)))
		else:
			print("ag%s = Prosumer(%s,ul%s)" % (str(i+1), str(i+1), str(i+1)))
	'''

	'''
	from agents import *
	'''

	'''
	ul1 = UserLoad()
	pv1 = PVGeneration(2.16)
	ag1 = Prosumer(1,pv1,ul1)
	ul2 = UserLoad()
	ag2 = Prosumer(2,ul2)
	ul3 = UserLoad()
	ag3 = Prosumer(3,ul3)
	ul4 = UserLoad()
	pv4 = PVGeneration(3.01)
	ag4 = Prosumer(4,pv4,ul4)
	ul5 = UserLoad()
	pv5 = PVGeneration(2.25)
	ag5 = Prosumer(5,pv5,ul5)
	ul6 = UserLoad()
	ag6 = Prosumer(6,ul6)
	ul7 = UserLoad()
	pv7 = PVGeneration(2.01)
	ag7 = Prosumer(7,pv7,ul7)
	ul8 = UserLoad()
	ag8 = Prosumer(8,ul8)
	ul9 = UserLoad()
	ag9 = Prosumer(9,ul9)
	ul10 = UserLoad()
	ag10 = Prosumer(10,ul10)
	ul11 = UserLoad()
	ag11 = Prosumer(11,ul11)
	ul12 = UserLoad()
	pv12 = PVGeneration(2.87)
	ag12 = Prosumer(12,pv12,ul12)
	ul13 = UserLoad()
	ag13 = Prosumer(13,ul13)
	ul14 = UserLoad()
	ag14 = Prosumer(14,ul14)
	ul15 = UserLoad()
	ag15 = Prosumer(15,ul15)
	ul16 = UserLoad()
	ag16 = Prosumer(16,ul16)
	ul17 = UserLoad()
	pv17 = PVGeneration(3.02)
	ag17 = Prosumer(17,pv17,ul17)
	ul18 = UserLoad()
	ag18 = Prosumer(18,ul18)
	ul19 = UserLoad()
	ag19 = Prosumer(19,ul19)
	ul20 = UserLoad()
	pv20 = PVGeneration(1.36)
	ag20 = Prosumer(20,pv20,ul20)
	ul21 = UserLoad()
	ag21 = Prosumer(21,ul21)
	ul22 = UserLoad()
	ag22 = Prosumer(22,ul22)
	ul23 = UserLoad()
	ag23 = Prosumer(23,ul23)
	ul24 = UserLoad()
	ag24 = Prosumer(24,ul24)
	ul25 = UserLoad()
	ag25 = Prosumer(25,ul25)
	ul26 = UserLoad()
	ag26 = Prosumer(26,ul26)
	ul27 = UserLoad()
	ag27 = Prosumer(27,ul27)
	ul28 = UserLoad()
	pv28 = PVGeneration(1.90)
	ag28 = Prosumer(28,pv28,ul28)
	ul29 = UserLoad()
	ag29 = Prosumer(29,ul29)
	ul30 = UserLoad()
	pv30 = PVGeneration(2.89)
	ag30 = Prosumer(30,pv30,ul30)
	ul31 = UserLoad()
	ag31 = Prosumer(31,ul31)
	ul32 = UserLoad()
	ag32 = Prosumer(32,ul32)
	ul33 = UserLoad()
	ag33 = Prosumer(33,ul33)
	ul34 = UserLoad()
	pv34 = PVGeneration(3.17)
	ag34 = Prosumer(34,pv34,ul34)
	ul35 = UserLoad()
	ag35 = Prosumer(35,ul35)
	ul36 = UserLoad()
	ag36 = Prosumer(36,ul36)
	ul37 = UserLoad()
	ag37 = Prosumer(37,ul37)
	ul38 = UserLoad()
	ag38 = Prosumer(38,ul38)
	ul39 = UserLoad()
	pv39 = PVGeneration(3.46)
	ag39 = Prosumer(39,pv39,ul39)
	ul40 = UserLoad()
	pv40 = PVGeneration(2.34)
	ag40 = Prosumer(40,pv40,ul40)
	ul41 = UserLoad()
	ag41 = Prosumer(41,ul41)
	ul42 = UserLoad()
	ag42 = Prosumer(42,ul42)
	ul43 = UserLoad()
	pv43 = PVGeneration(1.90)
	ag43 = Prosumer(43,pv43,ul43)
	ul44 = UserLoad()
	ag44 = Prosumer(44,ul44)
	ul45 = UserLoad()
	ag45 = Prosumer(45,ul45)
	ul46 = UserLoad()
	ag46 = Prosumer(46,ul46)
	ul47 = UserLoad()
	pv47 = PVGeneration(3.21)
	ag47 = Prosumer(47,pv47,ul47)
	ul48 = UserLoad()
	ag48 = Prosumer(48,ul48)
	ul49 = UserLoad()
	pv49 = PVGeneration(2.73)
	ag49 = Prosumer(49,pv49,ul49)
	ul50 = UserLoad()
	ag50 = Prosumer(50,ul50)
	ul51 = UserLoad()
	ag51 = Prosumer(51,ul51)
	ul52 = UserLoad()
	ag52 = Prosumer(52,ul52)
	ul53 = UserLoad()
	ag53 = Prosumer(53,ul53)
	ul54 = UserLoad()
	pv54 = PVGeneration(1.82)
	ag54 = Prosumer(54,pv54,ul54)
	ul55 = UserLoad()
	pv55 = PVGeneration(1.05)
	ag55 = Prosumer(55,pv55,ul55)
	ul56 = UserLoad()
	ag56 = Prosumer(56,ul56)
	ul57 = UserLoad()
	ag57 = Prosumer(57,ul57)
	ul58 = UserLoad()
	ag58 = Prosumer(58,ul58)
	ul59 = UserLoad()
	pv59 = PVGeneration(3.37)
	ag59 = Prosumer(59,pv59,ul59)
	ul60 = UserLoad()
	pv60 = PVGeneration(1.95)
	ag60 = Prosumer(60,pv60,ul60)
	ul61 = UserLoad()
	ag61 = Prosumer(61,ul61)
	ul62 = UserLoad()
	ag62 = Prosumer(62,ul62)
	ul63 = UserLoad()
	ag63 = Prosumer(63,ul63)
	ul64 = UserLoad()
	ag64 = Prosumer(64,ul64)
	ul65 = UserLoad()
	pv65 = PVGeneration(2.91)
	ag65 = Prosumer(65,pv65,ul65)
	ul66 = UserLoad()
	ag66 = Prosumer(66,ul66)
	ul67 = UserLoad()
	pv67 = PVGeneration(2.17)
	ag67 = Prosumer(67,pv67,ul67)
	ul68 = UserLoad()
	ag68 = Prosumer(68,ul68)
	ul69 = UserLoad()
	ag69 = Prosumer(69,ul69)
	ul70 = UserLoad()
	ag70 = Prosumer(70,ul70)
	ul71 = UserLoad()
	ag71 = Prosumer(71,ul71)
	ul72 = UserLoad()
	pv72 = PVGeneration(1.41)
	ag72 = Prosumer(72,pv72,ul72)
	ul73 = UserLoad()
	pv73 = PVGeneration(3.22)
	ag73 = Prosumer(73,pv73,ul73)
	ul74 = UserLoad()
	ag74 = Prosumer(74,ul74)
	ul75 = UserLoad()
	pv75 = PVGeneration(2.98)
	ag75 = Prosumer(75,pv75,ul75)
	ul76 = UserLoad()
	ag76 = Prosumer(76,ul76)
	ul77 = UserLoad()
	ag77 = Prosumer(77,ul77)
	ul78 = UserLoad()
	pv78 = PVGeneration(3.50)
	ag78 = Prosumer(78,pv78,ul78)
	ul79 = UserLoad()
	ag79 = Prosumer(79,ul79)
	ul80 = UserLoad()
	ag80 = Prosumer(80,ul80)
	ul81 = UserLoad()
	ag81 = Prosumer(81,ul81)
	ul82 = UserLoad()
	pv82 = PVGeneration(1.87)
	ag82 = Prosumer(82,pv82,ul82)
	ul83 = UserLoad()
	ag83 = Prosumer(83,ul83)
	ul84 = UserLoad()
	pv84 = PVGeneration(1.30)
	ag84 = Prosumer(84,pv84,ul84)
	ul85 = UserLoad()
	ag85 = Prosumer(85,ul85)
	ul86 = UserLoad()
	ag86 = Prosumer(86,ul86)
	ul87 = UserLoad()
	pv87 = PVGeneration(3.06)
	ag87 = Prosumer(87,pv87,ul87)
	ul88 = UserLoad()
	ag88 = Prosumer(88,ul88)
	ul89 = UserLoad()
	ag89 = Prosumer(89,ul89)
	ul90 = UserLoad()
	ag90 = Prosumer(90,ul90)
	ul91 = UserLoad()
	ag91 = Prosumer(91,ul91)
	ul92 = UserLoad()
	pv92 = PVGeneration(3.23)
	ag92 = Prosumer(92,pv92,ul92)
	ul93 = UserLoad()
	ag93 = Prosumer(93,ul93)
	ul94 = UserLoad()
	ag94 = Prosumer(94,ul94)
	ul95 = UserLoad()
	ag95 = Prosumer(95,ul95)
	ul96 = UserLoad()
	pv96 = PVGeneration(1.64)
	ag96 = Prosumer(96,pv96,ul96)
	ul97 = UserLoad()
	pv97 = PVGeneration(1.58)
	ag97 = Prosumer(97,pv97,ul97)
	ul98 = UserLoad()
	ag98 = Prosumer(98,ul98)
	ul99 = UserLoad()
	ag99 = Prosumer(99,ul99)
	ul100 = UserLoad()
	ag100 = Prosumer(100,ul100)

	ut = Utility()

	agr = Aggregator()

	start_time = '01/01/2019 - 00:00:00'
	finish_time = '15/02/2019 - 00:00:00'
	stochastic_time = '01/01/2019 - 23:45:00'
	step = 15

	controller = Controller(start_time, finish_time, stochastic_time, step, agr, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8, ag9, ag10,
	ag11, ag12, ag13, ag14, ag15, ag16, ag17, ag18, ag19, ag20,
	ag21, ag22, ag23, ag24, ag25, ag26, ag27, ag28, ag29, ag30,
	ag31, ag32, ag33, ag34, ag35, ag36, ag37, ag38, ag39, ag40, 
	ag41, ag42, ag43, ag44, ag45, ag46, ag47, ag48, ag49, ag50,
	ag51, ag52, ag53, ag54, ag55, ag56, ag57, ag58, ag59, ag60,
	ag61, ag62, ag63, ag64, ag65, ag66, ag67, ag68, ag69, ag70,
	ag71, ag72, ag73, ag74, ag75, ag76, ag77, ag78, ag79, ag80,
	ag81, ag82, ag83, ag84, ag85, ag86, ag87, ag88, ag89, ag90,
	ag91, ag92, ag93, ag94, ag95, ag96, ag97, ag98, ag99, ag100, ut)

	'''
