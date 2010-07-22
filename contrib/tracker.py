"""
Various classes for tracking attribute values. Work in progress.

$Id$
"""
__version__='$Revision: 8706 $'


# Note that the tracking classes don't depend on Topographica, but the
# tests do.


import param


def value_printer(**kw):
    obj_name = kw['obj'].name if hasattr(kw['obj'],'name') else str(kw['obj'])
    print "t=%s: %s.%s=%s"%(kw['time'],obj_name,kw['name'],kw['value'])


class Lister(object):
    def __init__(self,attrs):
        self.times = dict([(attr_name,list()) for attr_name in attrs])
        self.values = dict([(attr_name,list()) for attr_name in attrs])
        
    def __call__(self,**kw):
        self.times[kw['name']].append(kw['time'])
        self.values[kw['name']].append(kw['value'])


class Logger(param.Parameterized):

    # CB: need to do like plotfilesaver
    filename = param.String("test.txt")
        
    def __init__(self,attrs,**params):
        super(Logger,self).__init__(**params)
        self.f = open(self.filename,'w')
        
    def __call__(self,**kw):
        obj_name = kw['obj'].name if hasattr(kw['obj'],'name') else str(kw['obj'])
        log= "t=%s: %s.%s=%s\n"%(kw['time'],obj_name,kw['name'],kw['value'])
        self.f.write(log)

    def close(self):
        self.f.close()

# would be nice to have some that calculate summary statistics (e.g. running mean, variance, etc)
    

class AttributeTracker(param.Parameterized):
    """
    Tracks access to one or more of an object's attributes.


    By default, tracks every access to the specified attributes:

    >>> class Test(object):
    ...     name = 'test1'
    ...     def get_x(self):
    ...         return 1
    ...     x = property(get_x)

    >>> t = Test()
    >>> p = AttributeTracker(t,['x'])
    >>> junk=t.x
    t=None: test1.x=1
    >>> junk=t.x
    t=None: test1.x=1
    >>> junk=t.x
    t=None: test1.x=1
    

    Typical usage is to track a dynamic parameter of a Parameterized
    instance.  In that case, only track access once for any particular
    simulation time:

    >>> import topo
    >>> from topo import pattern
    >>> from topo.tests.utils import Series
    >>> g = pattern.Gaussian(x=Series(),name='testg')
    >>> p = AttributeTracker(g,['x'],time_fn=topo.sim.time)
    >>> junk=g()
    t=0: testg.x=0
    >>> junk=g()
    >>> # tracker records nothing because time didn't advance
    >>> topo.sim.run(1)
    >>> junk=g()
    t=1: testg.x=1


    Example of storing parameter values in memory:

    >>> g = pattern.Gaussian(x=Series(start=-1),y=Series(start=0),name='testg')
    >>> v = Lister(['x','y'])
    >>> p = AttributeTracker(g,['x','y'],time_fn=topo.sim.time,value_tracker=v)
    >>> junk=g()
    >>> topo.sim.run(1)
    >>> junk=g()
    >>> v.times['x']
    [mpq(1), mpq(2)]
    >>> v.values['x']
    [-1, 0]
    >>> v.times['y']
    [mpq(1), mpq(2)]
    >>> v.values['y']
    [0, 1]
    """

    value_tracker = param.Callable(value_printer)
    time_fn = param.Callable(None,constant=True,doc="""
    Set to topo.sim.time for tracking dynamic parameters in topographica simulations.""")

    
    def __init__(self,obj,attr_names,**params):
        super(AttributeTracker,self).__init__(**params)
        self.obj=obj

        # CEBALERT: if someone attaches multiple trackers to an
        # object, they have to be detached in reverse order
        # (should probably just prevent attaching multiple trackers)
        
        self._original_getattribute = type(obj).__getattribute__
        self._last_access= dict( [(attr_name,None) for attr_name in attr_names] )

        def _tracked_getattribute(instance,attr_name):
            v = self._original_getattribute(instance,attr_name) 
            t = self.time_fn() if self.time_fn is not None else None
            
            if instance is self.obj and attr_name in self._last_access and \
                   (self._last_access[attr_name] is None or t>self._last_access[attr_name]):
                self.value_tracker(obj=instance,name=attr_name,time=t,value=v)
                self._last_access[attr_name]=t
            return v

        type(obj).__getattribute__=_tracked_getattribute
                
        
    def stop_tracking(self):
        type(self.obj).__getattribute__=self._original_getattribute




## ## CB: currently unused

## class MethodTracker(param.Parameterized):
##     value_tracker = param.Callable(value_printer)
##     time_fn = param.Parameter(None,constant=True) # can't have Callable constant=True set to None

##     def __init__(self,obj,method_name,**params):
##         super(MethodTracker,self).__init__(**params)
##         self.obj = obj
##         self.method_name = method_name
##         self.original_method = getattr(type(obj),method_name)
##         self._last_time = None

##         def _tracked(instance,*args,**kw):
##             v = self.original_method(instance,*args,**kw)
##             t = self.time_fn() if self.time_fn is not None else None
    
##             if instance is self.obj and (self._last_time is None or t>self._last_time):
##                 self.value_tracker(self.method_name,t,v)
##                 self._last_time=t
##             return v

##         setattr(type(obj),self.method_name,_tracked)
                
##     def stop_tracking(self):
##         setattr(type(self.obj),self.method_name,self.original_method)
    


if __name__=='__main__' or __name__=="__mynamespace__": # for the ipython hack
    import doctest
    doctest.testmod(verbose=True)
