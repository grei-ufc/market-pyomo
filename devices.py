from random import uniform
import datetime as dt
#import enlopy as el
import numpy as np
import matplotlib.pyplot as plt
import random
import glb as glb
from stochastic import *
#import prosumer


'''
def generate_price_history(size):
	history = np.random.uniform(0.0,0.5,size)
	plt.plot(history)
	return history
'''

'''
datetimes = generate_timeseries('24/09/2018 - 00:00:00', 72*60*60, 5)
sl1 = ShiftableLoad('24/09/2018 - 00:00:00', '24/09/2018 - 10:00:00', 2.0, dt.timedelta(hours=0.5))
for datetime in datetimes:
	sl1.step(datetime)
'''


def generate_timeseries(start, time, step):
	'''
		start = string em formato datetime: dd/mm/YYYY - hh:mm:ss
		time = tamanho da série temporal em segundos
		step = step de tempo em minutos
	'''
	time_step = step * 60  # seconds
	dt_start = dt.datetime.strptime(start, '%d/%m/%Y - %H:%M:%S')
	delta = dt.timedelta(0, time)

	delta_sec = delta.days * (24 * 60 * 60) + delta.seconds

	res = [dt_start + dt.timedelta(0, t)
		   for t in range(0, delta_sec, time_step)]
	# res_pp = [i.strftime('%D - %T') for i in res]
	return res


# def marginal_cost(time_variable, constant_variable, delta_time):
#	return (time_variable*delta_time + constant_variable)

'''
datetimes = generate_timeseries('24/09/2018 - 00:00:00', 60*60, 5)
pv = PVGeneration(70)
for datetime in datetimes:
	print(datetime)
	pv.compute_curve()
'''


class PVGeneration(object):

	def __init__(self, max_power, option=None):
		self.type = 'PVGeneration'
		self.max_power = max_power
		self.curve = list()
		self.price_list = list()
		self.intended_daily_power = list()
		self.daily_power = list()
		if option is not None:
			print(option)
			if option <= 2:
				self.option = option
		else:
			self.option = random.randint(0, 2)

	def get_bid(self, time):
		self.curve = list()
		self.price_list = list()
		for i in range(glb.money_units * glb.money_maximum + 1):
			self.price_list.append(i / glb.money_units)
			self.curve.append(
				self.intended_daily_power[glb.pvgeneration_mapping[time.time().strftime("%H:%M")]])

		#plt.step(glb.price_list, self.curve,where='post')
		# print('plot!')
		return self.curve

	def data_setting(self, number):
		self.intended_daily_power = [-1 * self.max_power * i /
									 4 for i in glb.data_pv_generation[self.option][number]]

	def generate_scenario(self):
		scenarios = list()
		for i in range(20):
			create_scenario = list()
			for i in range(glb.contract_market_interval):
				create_scenario.extend(
					[-1 * self.max_power * i / 4 for i in random.choice(glb.data_pv_generation[self.option])])
			scenarios.append(create_scenario[:])
		probabilities = np.ones(len(scenarios))
		self.reduced_scenario = reduce_scenario(scenarios, probabilities, 3)
		# get n numbers of possible scenarios and append to a list
		# create a list of probabilities
		#self.reduced_scenario = reduce_scenario(scenarios, probabilities, 2)

	def set_power(self, price_position):
		self.power = self.curve[price_position]
		self.daily_power.append(self.power)
		return self.power

	def store_data(self, data, time, id_number):
		data.append([self.type, id_number, 'Energy (kWh)', time.strftime(
			'%d/%m/%Y')] + [i for i in self.daily_power[:]])
		self.daily_power = list()

	def store_total_data(self, total_data, id_number):
		pass

	def store_instantiation_data(self, instantiation_data, id_number):
		instantiation_data.append('pv%s = PVGeneration(%s,%s)'%(str(id_number),str(self.max_power),str(self.option)))

'''
datetimes = generate_timeseries('24/09/2018 - 00:00:00', 60*60, 5)
ul = UserLoad()
for datetime in datetimes:
	print(datetime)
	ul.compute_curve(uniform(20,50))
'''
#ul = UserLoad_curve()
# ul.compute_curve(50)


class UserLoad(object):

	def __init__(self, option=None):
		self.type = 'UserLoad'
		#self.max_power = max_power
		self.curve = list()
		self.price = list()
		self.all_prices = list()
		self.intended_daily_power = list()
		self.daily_power = list()
		self.power_db = list()
		self.my_percentile = random.randint(70, 90)
		self.id = None
		self.reduced_power = 0
		print(self.my_percentile)
		if option is not None:
			if option <= 14:
				self.option = option
		else:
			self.option = random.randint(0, 14)
		self.bidmask = glb.ul_bid_mask2

	def get_bid(self, time):
		self.curve = list()
		self.price_list = list()
		nominal_power = self.intended_daily_power[glb.userload_mapping[time.time(
		).strftime("%H:%M")]]
			
		#if self.all_prices == list():
		self.curve = [nominal_power * i for i in self.bidmask]
		
		'''
		else:
			start_decreasing = np.percentile(self.all_prices, self.my_percentile)
			
			self.bidmask = [glb.ul_variable_bid(x, start_decreasing, start_decreasing + self.my_percentile / 90, 0.15)
							for x in np.arange(0, glb.money_units * glb.money_maximum / 100, 0.01)]
			self.bidmask.append(2 * self.bidmask[-1] - self.bidmask[-2])
			
			self.curve = [nominal_power * i for i in glb.ul_bid_mask2]
			#if self.id == '1':
				#print("start_decreasing", start_decreasing)
				# plt.plot(self.bidmask)
		'''
		
		'''
		for i in range(glb.money_units+1):
			self.price_list.append(i/glb.money_units)
			self.curve.append(self.intended_daily_power[glb.userload_mapping[time.time().strftime("%H:%M")]])
		'''

		#plt.step(glb.price_list, self.curve,where='post')
		return self.curve

	def data_setting(self, number):
		self.intended_daily_power = [
			i / 2 for i in random.choice(glb.data_user_load[self.option])]

	def generate_scenario(self):
		scenarios = list()
		for i in range(20):
			create_scenario = list()
			for i in range(glb.contract_market_interval):
				create_scenario.extend(
					[i for i in random.choice(self.power_db)])
			scenarios.append(create_scenario[:])
		probabilities = np.ones(len(scenarios))
		try:
			self.reduced_scenario = reduce_scenario(
				scenarios, probabilities, 3)
		except:
			selected_scenario = list()
			selected_scenario.append(
				glb.contract_market_interval * random.choice(self.power_db))
			selected_scenario.append(
				glb.contract_market_interval * random.choice(self.power_db))
			selected_scenario.append(
				glb.contract_market_interval * random.choice(self.power_db))
			probabilities = [1 / 3, 1 / 3, 1 / 3]
			self.reduced_scenario = (selected_scenario, probabilities)
		'''
		scenarios = list()
		for i in range(20):
			create_scenario = list()
			for i in range(glb.contract_market_interval):
				create_scenario.extend([i/2 for i in random.choice(glb.data_user_load[self.option])])
			scenarios.append(create_scenario[:])
		probabilities = np.ones(len(scenarios))
		self.reduced_scenario = reduce_scenario(scenarios, probabilities, 3)
		'''

	def set_power(self, price_position):
		self.power = self.curve[price_position]
		self.daily_power.append(self.power)
		self.price.append(price_position / glb.money_units)
		self.reduced_power = self.reduced_power + self.curve[0] - self.power
		return self.power

	def store_data(self, data, time, id_number):
		if self.id == None:
			self.id = id_number
		self.power_db.append(self.daily_power[:])
		#print("tamanho do all_prices: ", len(self.all_prices))
		#print("tamanho do price: ", len(self.price))
		self.all_prices.extend(self.price[:])
		#print("tamanho do all_prices: ", len(self.all_prices))
		self.price = list()
		data.append([self.type, id_number, 'Energy (kWh)', time.strftime(
			'%d/%m/%Y')] + [i for i in self.daily_power[:]])
		self.daily_power = list()
		self.change_bid_mask()

	def change_bid_mask(self):

		start_decreasing = np.percentile(self.all_prices, self.my_percentile)
		self.bidmask = [glb.ul_variable_bid(x, start_decreasing, start_decreasing + self.my_percentile / 90, 0.15)
						for x in np.arange(0, glb.money_units * glb.money_maximum / 100, 0.01)]
		self.bidmask.append(2 * self.bidmask[-1] - self.bidmask[-2])
		

	def store_total_data(self, total_data, id_number):
		total_data.append([self.type, id_number, 'Total Reduced Energy (kWh)', self.reduced_power])

	def store_instantiation_data(self, instantiation_data, id_number):
		instantiation_data.append('pv%s = UserLoad(%s)'%(str(id_number),str(self.option)))

'''
datetimes = generate_timeseries('24/09/2018 - 00:00:00', 60*60, 5)
dg = DieselGeneration(0.8,2,10)
for datetime in datetimes:
	print(datetime)
	dg.state = dg.ON
	dg.compute_curve()
	dg.state = dg.OFF
	dg.compute_curve()
'''
#dg = DieselGeneration_curve(0.2,0.4,25)
# dg.compute_curve(1,1)


class Storage(object):

	def __init__(self, max_power):
		self.type = 'Storage'
		self.max_power = max_power
		self.curve = list()
		self.price_list = list()
		self.intended_daily_power = list()
		self.daily_power = list()

	def get_bid(self, time):
		self.curve = list()
		self.price_list = list()
		for i in range(glb.money_units + 1):
			self.price_list.append(i / glb.money_units)
			self.curve.append(
				-1 * self.intended_daily_power[glb.storage_mapping[time.time().strftime("%H:%M")]])

		#plt.step(glb.price_list, self.curve,where='post')
		# print('plot!')
		return self.curve

	def data_setting(self, number):
		self.intended_daily_power = [
			self.max_power * i for i in glb.data_storage]

	def generate_scenario(self):
		scenarios = list()
		for i in range(glb.contract_market_interval):
			scenarios.extend([self.max_power * i for i in glb.data_storage])
		probabilities = np.ones(len(scenarios))
		self.reduced_scenario = reduce_scenario(scenarios, probabilities, 1)
		# get n numbers of possible scenarios and append to a list
		# create a list of probabilities
		#self.reduced_scenario = reduce_scenario(scenarios, probabilities, 2)

	def set_power(self, price_position):
		self.power = self.curve[price_position]
		self.daily_power.append(self.power)
		return self.power

	def store_data(self, data, time, id_number):
		data.append([self.type, id_number, 'Energy (kWh)', time.strftime(
			'%d/%m/%Y')] + [i for i in self.daily_power[:]])
		self.daily_power = list()
