from sets import Set

class DB:
	
	def __init__(self,parent):
    	    self.children_params = []
	    self.children = []
	    self.data = {}
   	    self.register = []
	    
	    self.parent=parent
	    self.idd = self.register_yourself(self)
	    
	    
	def register_yourself(self,who):
	    if self.parent != None: return self.parent.register_yourself(who)
	    self.register.append(who)
	    return len(self.register)-1

		
	
	def get_child(self,param_dict):
	    for (cp,c) in zip(self.children_params,self.children):
	        if Set(param_dict.keys()) != Set(cp.keys()):
		   continue
		eq=True
		for k in cp.keys():
		    if cp[k] != param_dict[k]:
		       eq=False
	        if eq:
		   return c
	    self.children_params.append(param_dict)
	    new_node = DB(self)
	    self.children.append(new_node)
	    return new_node	
		
	def add_data(self,data_name,data,force=False):
	    if data_name in self.data.keys():
	       if force:	     
	       	  print "TRYING TO OVERWRITE DATA: ", data_name, " ALLOWING"
		  self.data[data_name] = data 
	       else:
		  print "TRYING TO OVERWRITE DATA: ", data_name, "NOT ALLOWING"
	    self.data[data_name] = data	
	
	def get_data(self,data_name):
	    if data_name in self.data.keys():
	       return self.data[data_name]
	    print "DATA DOES NOT EXIST:", data_name
	    return None 	

def loadResults(name):
    f = open(name,'rb')
    import pickle
    dd = pickle.load(f)
    f.close()
    return dd

def saveResults(dd,name):
    import pickle
    f = open(name,'wb')
    pickle.dump(dd,f,-2)
    f.close()