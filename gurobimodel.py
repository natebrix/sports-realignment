import gurobipy as gp

# wrapper for gurobipy model to enable solver-agnostic code.
class GurobiModel(gp.Model):    
    quicksum = gp.quicksum
    minimize = gp.GRB.MINIMIZE
    solver_name = 'gurobi'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.params.NonConvex = 0
        self.NumStart = 0
        self.constraint_callback = None

    def isNonconvex(self):
        return self.params.NonConvex == 2

    def setLogFile(self, filename):
        self.setParam(gp.GRB.Param.LogFile, filename)

    def addBinaryVar(self, name):
        return self.addVar(vtype=gp.GRB.BINARY, name=name) 

    def addIntegerVar(self, name, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY):
        return self.addVar(vtype=gp.GRB.INTEGER, name=name, lb=lb, ub=ub)

    def addContinuousVar(self, name, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY):
        return self.addVar(vtype=gp.GRB.CONTINUOUS, name=name, lb=lb, ub=ub)

    def setNonconvex(self, value):
        self.params.NonConvex = 2 if value else 0

    def setLazy(self, constraint, value):
        constraint.setAttr(gp.GRB.Attr.Lazy, value)

    def getVal(self, var):
        return var.X
    
    @property
    def name(self):
      return self.varName
    
    def createSol(self):
        s = self.NumStart
        self.NumStart += 1
        return s
    
    def setSolVal(self, solution, var, value):
        if self.params.StartNumber != solution:
            self.params.StartNumber = solution
            self.update()
        var.Start = value

    def addSol(self, solution):
        pass

    def setNonconvexSolVal(self, solution):
        pass

    def getSolvingTime(self):
        return self.getAttr(gp.GRB.Attr.Runtime)   
    
    def is_optimal(self):
        return self.status == gp.GRB.OPTIMAL
    
    def optimize(self):
        if self.constraint_callback:
            super().optimize(self.constraint_callback)
        else:
            super().optimize()

    def registerConstraintCallback(self, callback):
        def c(m, where):
            if where == gp.mip_callback:
                return callback(self)
            return False
        self.constraint_callback = c 

