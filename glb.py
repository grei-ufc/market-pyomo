import csv
import datetime as dt
import numpy as np

global fig

global data_user_load
data_user_load = list()
global data_pv_generation
data_pv_generation = list()
global data_storage
data_storage = list()
global contract_market_interval
contract_market_interval = 7
global stages_during_one_day
stages_during_one_day = 96
global tariffs_during_one_day
tariffs_during_one_day = 4

global tariff1_time
tariff1_time = dt.datetime.strptime('00:00:00', '%H:%M:%S').time()
global tariff2_time
tariff2_time = dt.datetime.strptime('06:00:00', '%H:%M:%S').time()
global tariff3_time
tariff3_time = dt.datetime.strptime('12:00:00', '%H:%M:%S').time()
global tariff4_time
tariff4_time = dt.datetime.strptime('18:00:00', '%H:%M:%S').time()
global tariff_mapping
tariff_mapping = {}

global userload_mapping
global pvgeneration_mapping
global storage_mapping
userload_mapping = dict()
pvgeneration_mapping = dict()
storage_mapping = dict()

global money_units
money_units = 100
global money_maximum
money_maximum = 2
global price_list
price_list = [(i / money_units) for i in range(money_units + 1)]

global stochastic_path
stochastic_path = '/home/talesmt/Stochastic/WorkSpace/Stochastic'


def logi(x, topo, base, centro, taxa):
    return -1 * (topo - base) * (1 / (1 + np.exp(-100 * taxa * (x - centro)))) + topo


def logip(x, topo, centro, taxa):
    return -1 * (topo) * (1 / (1 + np.exp(-100 * taxa * (x - centro))))


def cresc(x, topo, centro, taxa):
    return(-1 * (topo / (100 * taxa)) * np.log(np.exp((100 * taxa) * (x - centro)) + 1))

def decresc(topo, centro, taxa, money_range):
    centro = money_range - centro
    curve = [cresc(x, topo, centro, taxa)
                for x in np.arange(0, money_range, 0.01)]
    return [-i for i in curve[::-1]]

def ul_variable_bid(x, start_decreasing, stop_decreasing, reduction):
    taxa = 0.4
    return(1 - 1 * (((reduction) / (stop_decreasing - start_decreasing)) / (100 * taxa)) * np.log(np.exp((100 * taxa) * (x - start_decreasing)) + 1) +
           1 * (((reduction) / (stop_decreasing - start_decreasing)) / (100 * taxa)) * np.log(np.exp((100 * taxa) * (x - stop_decreasing)) + 1))


def ul_simplified_variable_bid(x, start_decreasing, stop_decreasing, reduction):
    taxa = 0.4
    return(((1 - reduction) / ((stop_decreasing - start_decreasing) * (100 * taxa))) * np.log((np.exp(x) + np.exp(stop_decreasing)) / (np.exp(x) + np.exp(start_decreasing))))


global ul_bid_mask2
ul_bid_mask2 = np.ones(money_units * money_maximum + 1)

base_price = 0.4
global base_price_tariff_1
base_price_tariff_1 = 1 * base_price
global base_price_tariff_2
base_price_tariff_2 = 2 * base_price
global base_price_tariff_3
base_price_tariff_3 = 3 * base_price
global base_price_tariff_4
base_price_tariff_4 = 4 * base_price


global contract_tariff_1
contract_tariff_1 = 0.9 * base_price_tariff_1
global contract_tariff_2
contract_tariff_2 = 0.9 * base_price_tariff_2
global contracte_tariff_3
contract_tariff_3 = 0.9 * base_price_tariff_3
global contract_tariff_4
contract_tariff_4 = 0.9 * base_price_tariff_4
global contract_tariffs
contract_tariffs = list()
contract_tariffs.append(contract_tariff_1)
contract_tariffs.append(contract_tariff_2)
contract_tariffs.append(contract_tariff_3)
contract_tariffs.append(contract_tariff_4)


#global ul_bid_mask
#ul_bid_mask = [logi(x,1,0.7,0.5,1) for x in np.arange(0,1,0.01)]
# ul_bid_mask.append(0.7)

#ut_bid_mask = [logip(x,400,0.5,0.3) for x in np.arange(0,1,0.01)]
# ut_bid_mask.append(400)
ut_bid_mask1_sell = [cresc(x, 200, base_price_tariff_1, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask1_sell.append(2 * ut_bid_mask1_sell[-1] - ut_bid_mask1_sell[-2])

ut_bid_mask1_buy = decresc(100, 0.8*contract_tariff_1, 0.5,money_maximum)
ut_bid_mask1_buy.append(2 * ut_bid_mask1_buy[-1] - ut_bid_mask1_buy[-2])

ut_bid_mask1 = [sum(x) for x in zip(ut_bid_mask1_sell, ut_bid_mask1_buy)]


ut_bid_mask1_sell = [cresc(x, 200, base_price_tariff_2, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask1_sell.append(2 * ut_bid_mask1_sell[-1] - ut_bid_mask1_sell[-2])

ut_bid_mask1_buy = decresc(100, 0.8*contract_tariff_2, 0.5,money_maximum)
ut_bid_mask1_buy.append(2 * ut_bid_mask1_buy[-1] - ut_bid_mask1_buy[-2])

ut_bid_mask2 = [sum(x) for x in zip(ut_bid_mask1_sell, ut_bid_mask1_buy)]


ut_bid_mask1_sell = [cresc(x, 200, base_price_tariff_3, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask1_sell.append(2 * ut_bid_mask1_sell[-1] - ut_bid_mask1_sell[-2])

ut_bid_mask1_buy = decresc(100, 0.8*contract_tariff_3, 0.5,money_maximum)
ut_bid_mask1_buy.append(2 * ut_bid_mask1_buy[-1] - ut_bid_mask1_buy[-2])

ut_bid_mask3 = [sum(x) for x in zip(ut_bid_mask1_sell, ut_bid_mask1_buy)]


ut_bid_mask1_sell = [cresc(x, 200, base_price_tariff_4, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask1_sell.append(2 * ut_bid_mask1_sell[-1] - ut_bid_mask1_sell[-2])

ut_bid_mask1_buy = decresc(100, 0.8*contract_tariff_4, 0.5,money_maximum)
ut_bid_mask1_buy.append(2 * ut_bid_mask1_buy[-1] - ut_bid_mask1_buy[-2])

ut_bid_mask4 = [sum(x) for x in zip(ut_bid_mask1_sell, ut_bid_mask1_buy)]

'''
ut_bid_mask2 = [cresc(x, 200, base_price_tariff_2, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask2.append(2 * ut_bid_mask2[-1] - ut_bid_mask2[-2])

ut_bid_mask3 = [cresc(x, 200, base_price_tariff_3, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask3.append(2 * ut_bid_mask3[-1] - ut_bid_mask3[-2])

ut_bid_mask4 = [cresc(x, 200, base_price_tariff_4, 0.5)
                for x in np.arange(0, money_units * money_maximum / 100, 1/money_units)]
ut_bid_mask4.append(2 * ut_bid_mask4[-1] - ut_bid_mask4[-2])
'''

global ut_bid_curves
ut_bid_curves = list()
ut_bid_curves.append(ut_bid_mask1)
ut_bid_curves.append(ut_bid_mask2)
ut_bid_curves.append(ut_bid_mask3)
ut_bid_curves.append(ut_bid_mask4)


global tariff_hours
tariff_hours = list()
tariff_hours.append(tariff1_time)
tariff_hours.append(tariff2_time)
tariff_hours.append(tariff3_time)
tariff_hours.append(tariff4_time)
#ut_bid_mask = [x/((1-0.55)*(0.4)) for x in ut_bid_mask]

'''
plt.plot([x for x in np.arange(0,1,0.01)],[logi(x,1,0.7,0.7,0.4) for x in np.arange(0,1,0.01)])
'''
'''
def load_data():
	path = "/home/talesmt/Stochastic/WorkSpace/Data/UserLoad/UserLoad_t.csv"
	with open(path, "r") as f:
		reader = csv.reader(f, delimiter=";", quoting=csv.QUOTE_NONNUMERIC)
		#global data_user_load
		data_user_load = list(reader)

	path = "/home/talesmt/Stochastic/WorkSpace/Data/PVGeneration/PVGeneration_t.csv"
	with open(path, "r") as f:
		reader = csv.reader(f, delimiter=";", quoting=csv.QUOTE_NONNUMERIC)
		#global data_pv_generation
		data_pv_generation = list(reader)

	path = "/home/talesmt/Stochastic/WorkSpace/Data/Storage/Storage_t.csv"
	with open(path, "r") as f:
		reader = csv.reader(f, delimiter=";", quoting=csv.QUOTE_NONNUMERIC)
		#global storage
		data_storage = list(reader)

	#global userload_mapping
	userload_mapping = {'23:45':95, '23:30':94, '23:15':93, '23:00':92, '22:45':91, '22:30':90, '22:15':89, '22:00':88, '21:45':87, '21:30':86, '21:15':85, '21:00':84, '20:45':83, '20:30':82, '20:15':81, '20:00':80, '19:45':79, '19:30':78, '19:15':77, '19:00':76, '18:45':75, '18:30':74, '18:15':73, '18:00':72, '17:45':71, '17:30':70, '17:15':69, '17:00':68, '16:45':67, '16:30':66, '16:15':65, '16:00':64, '15:45':63, '15:30':62, '15:15':61, '15:00':60, '14:45':59, '14:30':58, '14:15':57, '14:00':56, '13:45':55, '13:30':54, '13:15':53, '13:00':52, '12:45':51, '12:30':50, '12:15':49, '12:00':48, '11:45':47, '11:30':46, '11:15':45, '11:00':44, '10:45':43, '10:30':42, '10:15':41, '10:00':40, '09:45':39, '09:30':38, '09:15':37, '09:00':36, '08:45':35, '08:30':34, '08:15':33, '08:00':32, '07:45':31, '07:30':30, '07:15':29, '07:00':28, '06:45':27, '06:30':26, '06:15':25, '06:00':24, '05:45':23, '05:30':22, '05:15':21, '05:00':20, '04:45':19, '04:30':18, '04:15':17, '04:00':16, '03:45':15, '03:30':14, '03:15':13, '03:00':12, '02:45':11, '02:30':10, '02:15':9, '02:00':8, '01:45':7, '01:30':6, '01:15':5, '01:00':4, '00:45':3, '00:30':2, '00:15':1, '00:00':0}
	#global pvgeneration_mapping
	pvgeneration_mapping = {'23:45':95, '23:30':94, '23:15':93, '23:00':92, '22:45':91, '22:30':90, '22:15':89, '22:00':88, '21:45':87, '21:30':86, '21:15':85, '21:00':84, '20:45':83, '20:30':82, '20:15':81, '20:00':80, '19:45':79, '19:30':78, '19:15':77, '19:00':76, '18:45':75, '18:30':74, '18:15':73, '18:00':72, '17:45':71, '17:30':70, '17:15':69, '17:00':68, '16:45':67, '16:30':66, '16:15':65, '16:00':64, '15:45':63, '15:30':62, '15:15':61, '15:00':60, '14:45':59, '14:30':58, '14:15':57, '14:00':56, '13:45':55, '13:30':54, '13:15':53, '13:00':52, '12:45':51, '12:30':50, '12:15':49, '12:00':48, '11:45':47, '11:30':46, '11:15':45, '11:00':44, '10:45':43, '10:30':42, '10:15':41, '10:00':40, '09:45':39, '09:30':38, '09:15':37, '09:00':36, '08:45':35, '08:30':34, '08:15':33, '08:00':32, '07:45':31, '07:30':30, '07:15':29, '07:00':28, '06:45':27, '06:30':26, '06:15':25, '06:00':24, '05:45':23, '05:30':22, '05:15':21, '05:00':20, '04:45':19, '04:30':18, '04:15':17, '04:00':16, '03:45':15, '03:30':14, '03:15':13, '03:00':12, '02:45':11, '02:30':10, '02:15':9, '02:00':8, '01:45':7, '01:30':6, '01:15':5, '01:00':4, '00:45':3, '00:30':2, '00:15':1, '00:00':0}
	#global storage_mapping
	pvgeneration_mapping = {'23:45':95, '23:30':94, '23:15':93, '23:00':92, '22:45':91, '22:30':90, '22:15':89, '22:00':88, '21:45':87, '21:30':86, '21:15':85, '21:00':84, '20:45':83, '20:30':82, '20:15':81, '20:00':80, '19:45':79, '19:30':78, '19:15':77, '19:00':76, '18:45':75, '18:30':74, '18:15':73, '18:00':72, '17:45':71, '17:30':70, '17:15':69, '17:00':68, '16:45':67, '16:30':66, '16:15':65, '16:00':64, '15:45':63, '15:30':62, '15:15':61, '15:00':60, '14:45':59, '14:30':58, '14:15':57, '14:00':56, '13:45':55, '13:30':54, '13:15':53, '13:00':52, '12:45':51, '12:30':50, '12:15':49, '12:00':48, '11:45':47, '11:30':46, '11:15':45, '11:00':44, '10:45':43, '10:30':42, '10:15':41, '10:00':40, '09:45':39, '09:30':38, '09:15':37, '09:00':36, '08:45':35, '08:30':34, '08:15':33, '08:00':32, '07:45':31, '07:30':30, '07:15':29, '07:00':28, '06:45':27, '06:30':26, '06:15':25, '06:00':24, '05:45':23, '05:30':22, '05:15':21, '05:00':20, '04:45':19, '04:30':18, '04:15':17, '04:00':16, '03:45':15, '03:30':14, '03:15':13, '03:00':12, '02:45':11, '02:30':10, '02:15':9, '02:00':8, '01:45':7, '01:30':6, '01:15':5, '01:00':4, '00:45':3, '00:30':2, '00:15':1, '00:00':0}
	return
'''


def generate_timeseries(start, finish, interval):
    '''
        start = string em formato datetime: dd/mm/YYYY - hh:mm:ss
        finish = string em formato datetime: dd/mm/YYYY - hh:mm:ss
        interval = intervalo do evento em dias
    '''
    dt_start = dt.datetime.strptime(start, '%d/%m/%Y - %H:%M:%S')
    dt_finish = dt.datetime.strptime(finish, '%d/%m/%Y - %H:%M:%S')
    delta = dt.timedelta(interval)
    res = list()

    while dt_start < dt_finish:
        res.append(dt_start)
        dt_start = dt_start + delta

    return res


def oi():
    print('oi')
    return
