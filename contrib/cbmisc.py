"""
Miscellaneous code used by Chris B.
"""


def _create_single_generatorsheet_plotgroup(dim,sheet):
    import topo
    from topo.plotting.plotgroup import create_plotgroup
    from topo.command.analysis import measure_sine_pref
    pg= create_plotgroup(name='%s %s Preference'%(sheet,dim),category="Preference Maps",
                         doc='Measure preference for sine grating orientation.',
                         pre_plot_hooks=[measure_sine_pref.instance(sheet_views_prefix=sheet,generator_sheets=[topo.sim[sheet]])])
    pg.add_plot('%s %s Preference'%(sheet,dim),[('Hue','%s%sPreference'%(sheet,dim))])
    pg.add_plot('%s %s Preference&Selectivity'%(sheet,dim),
                [('Hue','%s%sPreference'%(sheet,dim)), 
                 ('Confidence','%s%sSelectivity'%(sheet,dim))])
    pg.add_plot('%s %s Selectivity'%(sheet,dim),[('Strength','%s%sSelectivity'%(sheet,dim))])
    pg.add_static_image('Color Key','command/or_key_white_vert_small.png')
    return pg

def add_one_eye_plotgroups():
    # Create Right and LeftRetina versions of Orientation Preference
    # plotgroup.
    import topo
    import copy
    import topo.plotting.plotgroup as P
    for sheet in ['LeftRetina','RightRetina']:
        for dim in ['Orientation']:#,'Hue']:
            pg = _create_single_generatorsheet_plotgroup(dim,sheet)
            P.plotgroups[pg.name]=pg

    if hasattr(topo,'guimain'):
        topo.guimain.refresh_plots_menu()


def analysis_function():
    import topo
    from topo.command.analysis import save_plotgroup
    from topo.command.basic import pattern_present,default_analysis_plotgroups
    from topo.base.projection import ProjectionSheet

    for pg in default_analysis_plotgroups:
        save_plotgroup(pg,use_cached_results=True)

    # Plot projections from each measured map
    measured_sheets = [s for s in topo.sim.objects(ProjectionSheet).values()
                       if hasattr(s,'measure_maps') and s.measure_maps]


    # CEBALERT: hack to ensure data has been measured for joint norm.
    for s in measured_sheets:
        for p in s.in_connections:            
            save_plotgroup("Projection",projection=p,normalize='single')

    for s in measured_sheets:
        for p in s.in_connections:
            # (equivalent to normalize='single' for non-jn'd projections)
            save_plotgroup("Projection",projection=p,normalize='joint')

    orcrod_analysis_function()

    # Test response to a standardized pattern
    from topo.pattern.basic import Gaussian
    from math import pi
    pattern_present(inputs=Gaussian(orientation=pi/4,aspect_ratio=4.7))
    save_plotgroup("Activity",saver_params={"filename_suffix":"_45d"})



def orcrod_analysis_function():
    # + L & R eye OR,CR maps
    import topo
    if ('LeftRetina' in topo.sim) and ('RightRetina' in topo.sim):
        from topo.command.analysis import save_plotgroup
        # (check the one-eye plotgroups have been created)
        import topo.plotting.plotgroup as P
        if 'LeftRetina Orientation Preference' not in P.plotgroups:
            add_one_eye_plotgroups()

        for s in ['LeftRetina','RightRetina']:
            save_plotgroup("%s Orientation Preference"%s,
                           use_cached_results=False)
            #save_plotgroup("%s Hue Preference"%s,
            #               use_cached_results=False)



######################################################################
######################################################################
## PLOTTING

import pylab
from topo.command.pylabplots import PylabPlotCommand

class polarplot(PylabPlotCommand):
    """
    pylab.polar plot with *supplied* theta values used for divisions
    on the theta axis (and optionally labeled with names from the
    'labels' list)
    
    e.g.
    bins = topo.sim['A'].data_analysis_fn.bins
    vals = topo.sim['A'].analysis_result
    polarplot(2*pi*(bins+0.05),vals,labels=bins)
    """

    # JABALERT: All but the first two arguments should probably be Parameters
    def __call__(self,theta_values,r_values,labels=None,fmt='%d',frac=1.1,**params):
        p = ParamOverrides(self,params)

        pylab.polar(theta_values,r_values)

        if labels is not None:
            # get grid lines at the theta_values specified with labels (rather than angles)
            # CB: why is 1st arg of pylab.thetagrids in degrees?
            pylab.thetagrids((360/(2*pi))*theta_values, labels=labels, fmt=fmt, frac=frac)

        self._generate_figure(p)

# CB: add polarhistogramplot(histogramplot)


class anyplot(PylabPlotCommand):
    """
    create a pylab plot using the topographica plot creation handling
    
    e.g.
    anyplot(pylab.contour,fn_args=(topo.sim['V1'].sheet_views['OrientationPreference'].view()[0],))
    """
    def __call__(self,fn,fn_args=None,fn_kw=None,**params):
        if fn_args is None:
            fn_args=tuple()
        if fn_kw is None:
            fn_kw=dict()
        assert isinstance(fn_args,tuple)
        assert isinstance(fn_kw,dict)

        p=ParamOverrides(self,params)
        pylab.figure(figsize=(5,5))
        fn(*fn_args,**fn_kw)
        self._generate_figure(p)








######################################################################
######################################################################
## ONLINE ANALYSIS


def get_activity(**data):
    """Helper function: return 'Activity' from data."""
    return data['Activity']

import numpy
import param

class DataAnalyser(param.Parameterized):
    """
    When called, accepts any list of keywords and returns XXXX numpy array
    according to data_transform_fn.
    """
    data_transform_fn = param.Parameter(default=get_activity,doc=
       """
       Gets the relevant data from those supplied.
       
       Will be called with data supplied to this class when the class
       itself is called; should return the data relevant for the
       analysis.
       """)

    def __call__(self,**data):
        relevant_data = self.data_transform_fn(**data)
        return relevant_data


class Summer(DataAnalyser):
    def __call__(self,**data):
        r = super(Summer,self).__call__(**data)
        return r.sum()


class Histogrammer(DataAnalyser):
    """DataAnalyser that returns a numpy.histogram."""

    num_bins = param.Number(10,constant=True,doc=
       """Number of bins for the histogram: see numpy.histogram.""")

    range_ = param.NumericTuple((0.0,1.0),constant=True,doc=
       """Range of data values: see numpy.histogram.""")

    def __init__(self,**params):
        super(Histogrammer,self).__init__(**params)
        self.bins = None

    def __call__(self,**data):
        d = super(Histogrammer,self).__call__(**data)        
        counts,self.bins = numpy.histogram(d,bins=self.num_bins,range=self.range_)
        return counts


from topo.base.simulation import EventProcessor
class OnlineAnalyser(EventProcessor):
    """
    EventProcessor that supplies data to a data_analysis_fn, then
    stores the result.

    If dest_ports is not None, it should specify a list of all data
    required before the data_analysis_fn is called (with that data).

    The result can be combined with previous ones by specifying an
    appropriate operator. 
    """
    data_analysis_fn = param.Callable(default=Summer(),constant=True,doc=
       """Callable to which the data are passed.""")
    
    operator_ = param.Parameter(default=None,doc=
       """
       Operator used to combine a result with previous results. If none, the current
       result overwrites the previous one.
       """)
    
    def __init__(self,dest_ports=None,**params):
        self.dest_ports = dest_ports
        super(OnlineAnalyser,self).__init__(**params)
        self.analysis_result = None
        self._data = {}
        
    def input_event(self,conn,data):

        self._data[conn.src_port]=data

        r = None

        if self.dest_ports is None or set(self._data.keys()).issuperset(set(self.dest_ports)):
            r = self.data_analysis_fn(**self._data)
            self._data={}

        if r is not None:
            if self.analysis_result is not None and self.operator_ is not None:
                self.analysis_result = self.operator_(r,self.analysis_result)
            else:
                self.analysis_result = r

