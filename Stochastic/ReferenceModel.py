from pyomo.core import *

#
# Model
#

model = AbstractModel()

#
# Parameters
#

model.I = Set()
model.T1 = Set()
model.T2 = Set()
model.T3 = Set()
model.T4 = Set()
model.PRICEPOOL = Param(model.I, within=NonNegativeReals)
model.PRICEDAYAHEADT1 = Param(within=PositiveReals)
model.PRICEDAYAHEADT2 = Param(within=PositiveReals)
model.PRICEDAYAHEADT3 = Param(within=PositiveReals)
model.PRICEDAYAHEADT4 = Param(within=PositiveReals)
model.DEMAND = Param(model.I)

#
# Variables
#

model.CONTRACTDAYAHEADT1 = Var(within=NonNegativeReals)
model.CONTRACTDAYAHEADT2 = Var(within=NonNegativeReals)
model.CONTRACTDAYAHEADT3 = Var(within=NonNegativeReals)
model.CONTRACTDAYAHEADT4 = Var(within=NonNegativeReals)
model.CONTRACTPOOL = Var(model.I, within=Reals)

#
# Constraints
#

def LimitPoolContract_rulet1(model, T1):
    if model.DEMAND[T1] >=0:
        return model.CONTRACTDAYAHEADT1 + model.CONTRACTPOOL[T1] >= model.DEMAND[T1]
    else:
        return model.CONTRACTPOOL[T1] >= model.DEMAND[T1]

model.LimitPoolContractt1 = Constraint(model.T1,rule=LimitPoolContract_rulet1)

def LimitPoolContract_rulet2(model, T2):
    if model.DEMAND[T2] >=0:
        return model.CONTRACTDAYAHEADT2 + model.CONTRACTPOOL[T2] >= model.DEMAND[T2]
    else:
        return model.CONTRACTPOOL[T2] >= model.DEMAND[T2]

model.LimitPoolContractt2 = Constraint(model.T2,rule=LimitPoolContract_rulet2)

def LimitPoolContract_rulet3(model, T3):
    if model.DEMAND[T3] >=0:
        return model.CONTRACTDAYAHEADT3 + model.CONTRACTPOOL[T3] >= model.DEMAND[T3]
    else:
        return model.CONTRACTPOOL[T3] >= model.DEMAND[T3]

model.LimitPoolContractt3 = Constraint(model.T3,rule=LimitPoolContract_rulet3)

def LimitPoolContract_rulet4(model, T4):
    if model.DEMAND[T4] >=0:
        return model.CONTRACTDAYAHEADT4 + model.CONTRACTPOOL[T4] >= model.DEMAND[T4]
    else:
        return model.CONTRACTPOOL[T4] >= model.DEMAND[T4]

model.LimitPoolContractt4 = Constraint(model.T4,rule=LimitPoolContract_rulet4)

def LimitPoolContract_rulet5(model, i):
    if model.DEMAND[i] >= 0:
        return model.CONTRACTPOOL[i] >= 0
    else:
        return model.CONTRACTPOOL[i] <= 0

model.LimitPoolContractt5 = Constraint(model.I,rule=LimitPoolContract_rulet5)


#
# Stage-specific cost computations
#

def Stagecost_rule(model):
    return model.CONTRACTDAYAHEADT1

model.StageCost = Expression(rule=Stagecost_rule)


def Objective_rule(model):
    return (model.CONTRACTDAYAHEADT1*model.PRICEDAYAHEADT1 + 
    	model.CONTRACTDAYAHEADT2*model.PRICEDAYAHEADT2 + 
    	model.CONTRACTDAYAHEADT3*model.PRICEDAYAHEADT3 +
    	model.CONTRACTDAYAHEADT4*model.PRICEDAYAHEADT4 +
    	summation(model.PRICEPOOL,model.CONTRACTPOOL))

model.OBJECTIVE = Objective(rule=Objective_rule,sense=minimize)
