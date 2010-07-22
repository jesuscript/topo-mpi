# Work in progress.
#
# Code from ExpressionParameter would be merged in to Dynamic to allow
# any Dynamic parameter to contain an expression (and so expressions
# would obey any time_fn).

import param
from param.parameterized import ParamOverrides
import types
import inspect

class ExpressionParameter(param.Parameter):
    """
    Parameter that can be set to an expression that is evaluated each
    time its value is requested.
    """
    __slots__ = ['initial_value']

    def __init__(self,expression,initial_value=None,**kw):
        self.initial_value = initial_value
        super(ExpressionParameter,self).__init__(default=expression,**kw)
        
    def __get__(self,obj,objtype):
        val = super(ExpressionParameter,self).__get__(obj,objtype)

        if isinstance(val,types.FunctionType) and inspect.getargs(val.func_code)==(['p'],None,None):
        
            if not obj:
                # called on class
                my_val = self.initial_value
                O = objtype
            else:
                # called on instance; get last value produced for obj
                my_val = obj.__dict__.get(self._internal_name+"_last",self.initial_value)
                O = obj

            new_value = val(ParamOverrides(O,{self._attrib_name:my_val}))

            # store the generated value
            if obj:
                setattr(obj,self._internal_name+"_last",new_value)
            else:
                self.initial_value = new_value

            return new_value

        else:
            return val



if __name__=='__mynamespace__' or __name__=='__main__':
    from math import pi
    from topo import pattern
    
    class Gaussian2(pattern.Gaussian):
        v = param.Parameter(1.0)
        x = ExpressionParameter(lambda p: p.x+p.v*pi, initial_value=0)

    g2 = Gaussian2()


