import numpy 
import __main__

def complex_analysis_function():
    """
    Basic example of an analysis command for run_batch; users are
    likely to need something similar but highly customized.
    """
    import topo
    from topo.command.analysis import save_plotgroup
    from topo.base.projection import ProjectionSheet
    from topo.sheet.basic import GeneratorSheet
    from topo.analysis.featureresponses import SinusoidalMeasureResponseCommand,FeatureCurveCommand
    import contrib.jacommands
    exec "from topo.analysis.vision import analyze_complexity" in __main__.__dict__
    
    print 'Analysing'
    
    import matplotlib
    matplotlib.rc('xtick', labelsize=17)
    matplotlib.rc('ytick', labelsize=17)					
			
    # Build a list of all sheets worth measuring
    f = lambda x: hasattr(x,'measure_maps') and x.measure_maps
    measured_sheets = filter(f,topo.sim.objects(ProjectionSheet).values())
    input_sheets = topo.sim.objects(GeneratorSheet).values()
							    
    # Set potentially reasonable defaults; not necessarily useful
    topo.command.analysis.coordinate=(0.0,0.0)
    if input_sheets:    topo.command.analysis.input_sheet_name=input_sheets[0].name
    if measured_sheets: topo.command.analysis.sheet_name=measured_sheets[0].name
    
    FeatureCurveCommand.curve_parameters=[{"contrast":30},{"contrast":50},{"contrast":70},{"contrast":90}]
    
    import numpy
    # reset treshold and desable noise before measuring maps
    #m = numpy.mean(topo.sim["V1Simple"].output_fns[2].t)
    #topo.sim["V1Simple"].output_fns[2].t*=0
    #topo.sim["V1Simple"].output_fns[2].t+=m
    #sc = topo.sim["V1Simple"].output_fns[1].generator.scale
    #topo.sim["V1Simple"].output_fns[1].generator.scale=0.0
    a = topo.sim["V1Complex"].in_connections[0].strength
    
    SinusoidalMeasureResponseCommand.scale=__main__.__dict__.get("analysis_scale",0.35)


    if((float(topo.sim.time()) >= 5003.0) and (float(topo.sim.time()) < 5004.0)): 
	topo.sim["V1Complex"].in_connections[0].strength=0
	SinusoidalMeasureResponseCommand.frequencies=[3.0]

    if((float(topo.sim.time()) >= 5005.0) and (float(topo.sim.time()) < 5006.0)): 
	SinusoidalMeasureResponseCommand.frequencies=[3.0]

    if((float(topo.sim.time()) >= 5006.0) and (float(topo.sim.time()) < 5007.0)): 
    	topo.sim["V1Complex"].in_connections[0].strength=0
	SinusoidalMeasureResponseCommand.frequencies=[2.4]

    if((float(topo.sim.time()) >= 5007.0) and (float(topo.sim.time()) < 5008.0)): 
	SinusoidalMeasureResponseCommand.frequencies=[2.4]



    if((float(topo.sim.time()) >= 10002.0) and (float(topo.sim.time()) < 10003.0)): 
	topo.sim["V1Complex"].in_connections[0].strength=0
	SinusoidalMeasureResponseCommand.frequencies=[2.4]

    if((float(topo.sim.time()) >= 10003.0) and (float(topo.sim.time()) < 10004.0)): 
	topo.sim["V1Complex"].in_connections[0].strength=0
	SinusoidalMeasureResponseCommand.frequencies=[3.0]

    if((float(topo.sim.time()) >= 10004.0) and (float(topo.sim.time()) < 10005.0)): 
	SinusoidalMeasureResponseCommand.frequencies=[2.4]

    if((float(topo.sim.time()) >= 10005.0) and (float(topo.sim.time()) < 10006.0)): 
	SinusoidalMeasureResponseCommand.frequencies=[3.0]


    save_plotgroup("Orientation Preference and Complexity")
    save_plotgroup("Activity")

									
    # Plot all projections for all measured_sheets
    for s in measured_sheets:
        for p in s.projections().values():
            save_plotgroup("Projection",projection=p)

    
    if(float(topo.sim.time()) >= 10005.0): 
        print 'Measuring orientations'
        SinusoidalMeasureResponseCommand.frequencies=[2.4]
        topo.command.pylabplots.measure_or_tuning_fullfield.instance(sheet=topo.sim["V1Complex"])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1Complex"],coords=[(0,0)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0.1]",sheet=topo.sim["V1Complex"],coords=[(0.1,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,-0.1]",sheet=topo.sim["V1Complex"],coords=[(0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0.1]",sheet=topo.sim["V1Complex"],coords=[(-0.1,0.1)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,-0.1]",sheet=topo.sim["V1Complex"],coords=[(-0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,0.2]",sheet=topo.sim["V1Complex"],coords=[(0.2,0.2)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,-0.2]",sheet=topo.sim["V1Complex"],coords=[(0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,0.2]",sheet=topo.sim["V1Complex"],coords=[(-0.2,0.2)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,-0.2]",sheet=topo.sim["V1Complex"],coords=[(-0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.1]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.1]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0]",sheet=topo.sim["V1Complex"],coords=[(-0.1,0.0)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0]",sheet=topo.sim["V1Complex"],coords=[(0.1,-0.0)])()

        topo.command.pylabplots.measure_or_tuning_fullfield.instance(sheet=topo.sim["V1Simple"])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,0]",sheet=topo.sim["V1Simple"],coords=[(0,0)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.1,0.1]",sheet=topo.sim["V1Simple"],coords=[(0.1,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.1,-0.1]",sheet=topo.sim["V1Simple"],coords=[(0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.1,0.1]",sheet=topo.sim["V1Simple"],coords=[(-0.1,0.1)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.1,-0.1]",sheet=topo.sim["V1Simple"],coords=[(-0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.2,0.2]",sheet=topo.sim["V1Simple"],coords=[(0.2,0.2)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.2,-0.2]",sheet=topo.sim["V1Simple"],coords=[(0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.2,0.2]",sheet=topo.sim["V1Simple"],coords=[(-0.2,0.2)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.2,-0.2]",sheet=topo.sim["V1Simple"],coords=[(-0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,0.1]",sheet=topo.sim["V1Simple"],coords=[(0.0,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,-0.1]",sheet=topo.sim["V1Simple"],coords=[(0.0,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.1,0]",sheet=topo.sim["V1Simple"],coords=[(-0.1,0.0)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.1,0]",sheet=topo.sim["V1Simple"],coords=[(0.1,-0.0)])()

    #topo.sim["V1Simple"].output_fns[1].generator.scale=sc
    topo.sim["V1Complex"].in_connections[0].strength = a
    #topo.sim["V1Complex"].in_connections[0].strength=st

def complex_surround_analysis_function():

    """
    Basic example of an analysis command for run_batch; users are
    likely to need something similar but highly customized.
    """
    import topo
    from topo.command.analysis import save_plotgroup
    from topo.analysis.featureresponses import SinusoidalMeasureResponseCommand,FeatureCurveCommand
    from topo.base.projection import ProjectionSheet
    from topo.sheet.basic import GeneratorSheet
    import contrib.jacommands
    import contrib.surround_analysis
    exec "from topo.analysis.vision import analyze_complexity" in __main__.__dict__

    import matplotlib
    matplotlib.rc('xtick', labelsize=17)
    matplotlib.rc('ytick', labelsize=17)					
					
			
    SinusoidalMeasureResponseCommand.frequencies=[3.0]    
    SinusoidalMeasureResponseCommand.scale=__main__.__dict__.get("analysis_scale",0.3)
    from topo.analysis.featureresponses import PatternPresenter            
    PatternPresenter.duration=2.0
    import topo.command.pylabplots
    reload(topo.command.pylabplots)

    # Build a list of all sheets worth measuring
    f = lambda x: hasattr(x,'measure_maps') and x.measure_maps
    measured_sheets = filter(f,topo.sim.objects(ProjectionSheet).values())
    input_sheets = topo.sim.objects(GeneratorSheet).values()
							    
    # Set potentially reasonable defaults; not necessarily useful
    topo.command.analysis.coordinate=(0.0,0.0)
    if input_sheets:    topo.command.analysis.input_sheet_name=input_sheets[0].name
    if measured_sheets: topo.command.analysis.sheet_name=measured_sheets[0].name
									    
    save_plotgroup("Orientation Preference and Complexity")
    save_plotgroup("Activity",normalize=True)
										
    # Plot all projections for all measured_sheets
    for s in measured_sheets:
        for p in s.projections().values():
            save_plotgroup("Projection",projection=p)
    

    if(float(topo.sim.time()) > 6020.0): 
        contrib.surround_analysis.run_dynamics_analysis(0.0,0.0,0.7,__main__.__dict__.get("analysis_scale",0.3))
        topo.command.pylabplots.measure_or_tuning_fullfield.instance(sheet=topo.sim["V1Complex"])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1Complex"],coords=[(0,0)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0.1]",sheet=topo.sim["V1Complex"],coords=[(0.1,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,-0.1]",sheet=topo.sim["V1Complex"],coords=[(0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0.1]",sheet=topo.sim["V1Complex"],coords=[(-0.1,0.1)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,-0.1]",sheet=topo.sim["V1Complex"],coords=[(-0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,0.2]",sheet=topo.sim["V1Complex"],coords=[(0.2,0.2)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,-0.2]",sheet=topo.sim["V1Complex"],coords=[(0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,0.2]",sheet=topo.sim["V1Complex"],coords=[(-0.2,0.2)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,-0.2]",sheet=topo.sim["V1Complex"],coords=[(-0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.1]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.1]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0]",sheet=topo.sim["V1Complex"],coords=[(-0.1,0.0)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0]",sheet=topo.sim["V1Complex"],coords=[(0.1,-0.0)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.3,0.3]",sheet=topo.sim["V1Complex"],coords=[(0.3,0.3)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.3,-0.3]",sheet=topo.sim["V1Complex"],coords=[(0.3,-0.3)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.3,0.3]",sheet=topo.sim["V1Complex"],coords=[(-0.3,0.3)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.3,-0.3]",sheet=topo.sim["V1Complex"],coords=[(-0.3,-0.3)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.24,0.24]",sheet=topo.sim["V1Complex"],coords=[(0.24,0.24)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.24,-0.24]",sheet=topo.sim["V1Complex"],coords=[(0.24,-0.24)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.24,0.24]",sheet=topo.sim["V1Complex"],coords=[(-0.24,0.42)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.24,-0.24]",sheet=topo.sim["V1Complex"],coords=[(-0.24,-0.24)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.24]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.24)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.24]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.42)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.24,0]",sheet=topo.sim["V1Complex"],coords=[(-0.24,0.0)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.24,0]",sheet=topo.sim["V1Complex"],coords=[(0.24,-0.0)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.3]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.3)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.3]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.3)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.3,0]",sheet=topo.sim["V1Complex"],coords=[(-0.3,0.0)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.3,0]",sheet=topo.sim["V1Complex"],coords=[(0.3,-0.0)])()

        #topo.command.pylabplots.measure_or_tuning_fullfield.instance(sheet=topo.sim["V1Simple"])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,0]",sheet=topo.sim["V1Simple"],coords=[(0,0)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.1,0.1]",sheet=topo.sim["V1Simple"],coords=[(0.1,0.1)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.1,-0.1]",sheet=topo.sim["V1Simple"],coords=[(0.1,-0.1)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.1,0.1]",sheet=topo.sim["V1Simple"],coords=[(-0.1,0.1)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.1,-0.1]",sheet=topo.sim["V1Simple"],coords=[(-0.1,-0.1)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.2,0.2]",sheet=topo.sim["V1Simple"],coords=[(0.2,0.2)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.2,-0.2]",sheet=topo.sim["V1Simple"],coords=[(0.2,-0.2)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.2,0.2]",sheet=topo.sim["V1Simple"],coords=[(-0.2,0.2)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.2,-0.2]",sheet=topo.sim["V1Simple"],coords=[(-0.2,-0.2)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,0.1]",sheet=topo.sim["V1Simple"],coords=[(0.0,0.1)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,-0.1]",sheet=topo.sim["V1Simple"],coords=[(0.0,-0.1)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.1,0]",sheet=topo.sim["V1Simple"],coords=[(-0.1,0.0)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.1,0]",sheet=topo.sim["V1Simple"],coords=[(0.1,-0.0)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.3,0.3]",sheet=topo.sim["V1Simple"],coords=[(0.3,0.3)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.3,-0.3]",sheet=topo.sim["V1Simple"],coords=[(0.3,-0.3)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.3,0.3]",sheet=topo.sim["V1Simple"],coords=[(-0.3,0.3)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.3,-0.3]",sheet=topo.sim["V1Simple"],coords=[(-0.3,-0.3)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.24,0.24]",sheet=topo.sim["V1Simple"],coords=[(0.24,0.24)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.24,-0.24]",sheet=topo.sim["V1Simple"],coords=[(0.24,-0.24)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.24,0.24]",sheet=topo.sim["V1Simple"],coords=[(-0.24,0.42)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.24,-0.24]",sheet=topo.sim["V1Simple"],coords=[(-0.24,-0.24)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,0.24]",sheet=topo.sim["V1Simple"],coords=[(0.0,0.24)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,-0.24]",sheet=topo.sim["V1Simple"],coords=[(0.0,-0.42)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.24,0]",sheet=topo.sim["V1Simple"],coords=[(-0.24,0.0)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.24,0]",sheet=topo.sim["V1Simple"],coords=[(0.24,-0.0)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,0.3]",sheet=topo.sim["V1Simple"],coords=[(0.0,0.3)])()
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0,-0.3]",sheet=topo.sim["V1Simple"],coords=[(0.0,-0.3)])()
	#topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[-0.3,0]",sheet=topo.sim["V1Simple"],coords=[(-0.3,0.0)])()    
        #topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="SimpleORTC[0.3,0]",sheet=topo.sim["V1Simple"],coords=[(0.3,-0.0)])()

        contrib.surround_analysis.surround_analysis("V1Complex").analyse([(0,0),(5,0),(-5,0),(0,5),(0,-5),(5,5),(5,-5),(-5,5),(-5,-5),(8,0),(-8,0),(0,8),(0,-8),(8,8),(8,-8),(-8,8),(-8,-8)],15,5)


													    
def v2_analysis_function():
    """
    Basic example of an analysis command for run_batch; users are
    likely to need something similar but highly customized.
    """
    import topo
    from topo.command.analysis import save_plotgroup
    from topo.base.projection import ProjectionSheet
    from topo.sheet.basic import GeneratorSheet
    exec "from topo.analysis.vision import analyze_complexity" in __main__.__dict__
    from param import normalize_path

    topo.sim["V1Simple"].measure_maps = True
    topo.sim["V1Complex"].measure_maps = True
    topo.sim["V2"].measure_maps = True
    
    topo.sim["V2"].in_connections[0].strength=4
    
    save_plotgroup("Orientation Preference and Complexity")    

    # Plot all projections for all measured_sheets
    measured_sheets = [s for s in topo.sim.objects(ProjectionSheet).values()
                       if hasattr(s,'measure_maps') and s.measure_maps]
    for s in measured_sheets:
        for p in s.projections().values():
            save_plotgroup("Projection",projection=p)

    save_plotgroup("Activity")
#    topo.sim["V1Simple"].measure_maps = False
#    topo.sim["V1Complex"].measure_maps = False
        
    save_plotgroup("Corner OR Preference")
    from topo.command.basic import save_snapshot
#    save_snapshot(normalize_path('snapshot.typ'))


activity_history=numpy.array([])
def rf_analysis():
    import topo
    import pylab
    import topo.analysis.vision
    import contrib.jacommands
    from topo.command.analysis import save_plotgroup
    from topo.base.projection import ProjectionSheet
    from topo.sheet.basic import GeneratorSheet
    from topo.command.analysis import measure_or_tuning_fullfield, measure_or_pref
    from topo.command.pylabplots import cyclic_tuning_curve
    from param import normalize_path    
    
    if(float(topo.sim.time()) <=20010): 
        save_plotgroup("Orientation Preference")
        save_plotgroup("Activity")
    
        # Plot all projections for all measured_sheets
        measured_sheets = [s for s in topo.sim.objects(ProjectionSheet).values()
                           if hasattr(s,'measure_maps') and s.measure_maps]
        for s in measured_sheets:
            for p in s.projections().values():
                save_plotgroup("Projection",projection=p)

        prefix="WithGC"   
        measure_or_tuning_fullfield()
        s=topo.sim["V1"]
        cyclic_tuning_curve(filename_suffix=prefix,filename="OrientationTC:V1:[0,0]",sheet=s,coords=[(0,0)],x_axis="orientation")
        cyclic_tuning_curve(filename_suffix=prefix,filename="OrientationTC:V1:[0.1,0.1]",sheet=s,coords=[(0.1,0.1)],x_axis="orientation")
        cyclic_tuning_curve(filename_suffix=prefix,filename="OrientationTC:V1:[-0.1,-0.1]",sheet=s,coords=[(-0.1,-0.1)],x_axis="orientation")
        cyclic_tuning_curve(filename_suffix=prefix,filename="OrientationTC:V1:[0.1,-0.1]",sheet=s,coords=[(0.1,-0.1)],x_axis="orientation")
        cyclic_tuning_curve(filename_suffix=prefix,filename="OrientationTC:V1:[-0.1,0.1]",sheet=s,coords=[(-0.1,0.1)],x_axis="orientation")
    else:
        topo.command.basic.activity_history = numpy.concatenate((contrib.jacommands.activity_history,topo.sim["V1"].activity.flatten()),axis=1)    

    if(float(topo.sim.time()) == 20000): 
        topo.sim["V1"].plastic=False
        contrib.jacommands.homeostatic_analysis_function()

    if(float(topo.sim.time()) == 20001): 
        pylab.figure()

def gc_homeo_af():
    import contrib.jsldefs
    import topo.command.pylabplots
    import contrib.jacommands
    from topo.command.analysis import save_plotgroup
    from topo.analysis.featureresponses import FeatureResponses , PatternPresenter, FeatureMaps            
    #FeatureResponses.repetitions=10

    FeatureMaps.selectivity_multiplier=20

    PatternPresenter.duration=0.2
    PatternPresenter.apply_output_fns=False
    import topo.command.pylabplots
    reload(topo.command.pylabplots)

    
    on = topo.sim["LGNOn"].in_connections[0].strength
    off = topo.sim["LGNOff"].in_connections[0].strength
    if __main__.__dict__.get("GC",False):
       topo.sim["LGNOn"].in_connections[0].strength=0
       topo.sim["LGNOff"].in_connections[0].strength=0
    
    contrib.jsldefs.homeostatic_analysis_function()
    topo.command.pylabplots.fftplot(topo.sim["V1"].sheet_views["OrientationPreference"].view()[0],filename="V1ORMAPFFT")
    
    from topo.misc.filepath import normalize_path, application_path    
    from scipy.io import write_array
    import numpy
    write_array(normalize_path(str(topo.sim.time())+"orprefmap.txt"), topo.sim["V1"].sheet_views["OrientationPreference"].view()[0])
    write_array(normalize_path(str(topo.sim.time())+"orselmap.txt"), topo.sim["V1"].sheet_views["OrientationSelectivity"].view()[0])
    topo.sim["LGNOn"].in_connections[0].strength = on
    topo.sim["LGNOff"].in_connections[0].strength = off

    print float(topo.sim.time())
    if(float(topo.sim.time()) > 19002.0): 
	#topo.sim["V1"].output_fns[2].scale=0.0
	save_plotgroup("Position Preference")
	PatternPresenter.duration=1.0
        PatternPresenter.apply_output_fns=True
	import topo.command.pylabplots
        reload(topo.command.pylabplots)
        topo.command.pylabplots.measure_or_tuning_fullfield.instance(sheet=topo.sim["V1"],repetitions=10)(repetitions=10)
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1"],coords=[(0,0)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1"],coords=[(0.1,0)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1"],coords=[(0.1,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1"],coords=[(0,0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0.1]",sheet=topo.sim["V1"],coords=[(0.1,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,-0.1]",sheet=topo.sim["V1"],coords=[(0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0.1]",sheet=topo.sim["V1"],coords=[(-0.1,0.1)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,-0.1]",sheet=topo.sim["V1"],coords=[(-0.1,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,0.2]",sheet=topo.sim["V1"],coords=[(0.2,0.2)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,-0.2]",sheet=topo.sim["V1"],coords=[(0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,0.2]",sheet=topo.sim["V1"],coords=[(-0.2,0.2)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,-0.2]",sheet=topo.sim["V1"],coords=[(-0.2,-0.2)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.1]",sheet=topo.sim["V1"],coords=[(0.0,0.1)])()
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.1]",sheet=topo.sim["V1"],coords=[(0.0,-0.1)])()
	topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0]",sheet=topo.sim["V1"],coords=[(-0.1,0.0)])()    
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0]",sheet=topo.sim["V1"],coords=[(0.1,-0.0)])()

    if(float(topo.sim.time()) > 20000.0): 
        topo.sim["V1"].output_fns[1].plastic=False
        contrib.jacommands.measure_histogram(iterations=1000) 	
    

def saver_function():
    from topo.command.basic import save_snapshot
    save_snapshot(normalize_path('snapshot.typ'))

def empty():
    a = 10
