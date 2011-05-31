import param

import topo.pattern.basic
import topo.pattern.random
import __main__
import os
import contrib
import topo
from topo.transferfn.misc import PatternCombine
from topo.transferfn.misc import HalfRectify
from topo import numbergen
from topo.pattern.basic import Gaussian
from topo.numbergen import UniformRandom, BoundedNumber, ExponentialDecay
from topo.command.basic import pattern_present
from param import normalize_path
import numpy

from topo.analysis.featureresponses import MeasureResponseCommand, FeatureMaps, SinusoidalMeasureResponseCommand, FeatureCurveCommand
FeatureMaps.num_orientation=16
MeasureResponseCommand.scale=1.0
MeasureResponseCommand.duration=4.0
SinusoidalMeasureResponseCommand.frequencies=[2.4]
FeatureCurveCommand.num_orientation=16
FeatureCurveCommand.curve_parameters=[{"contrast":15},{"contrast":50},{"contrast":90}]
from topo.command.basic import load_snapshot


load_snapshot('CCSimple_010005_new_or_map.00.typ')
    
from topo.command.basic import wipe_out_activity, clear_event_queue
wipe_out_activity()
clear_event_queue()

from topo.pattern.basic import SineGrating, Disk
class SineGratingDiskTemp(SineGrating):
      mask_shape = param.Parameter(default=Disk(smoothing=0,size=1.0))

def set_parameters(a,b,c,d,e,f,g,h,i,j,k,l):
    print a,b,c,d,e,f,g,h,i,j,k,l
    
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackExc1"].strength=b
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackInh"].strength=c
    topo.sim["V1Complex"].projections()["LongEE"].strength=d
    topo.sim["V1ComplexInh"].projections()["LongEI"].strength=e
    topo.sim["V1Complex"].projections()["LocalIE"].strength=f
    topo.sim["V1ComplexInh"].projections()["LocalII"].strength=g
    topo.sim["V1Complex"].projections()["V1SimpleAfferent"].strength=h
    topo.sim["V1Complex"].projections()["LocalEE"].strength=i
    topo.sim["V1ComplexInh"].projections()["LocalEI"].strength=j
    topo.sim["V1Complex"].output_fns[1].t*=0    
    topo.sim["V1Complex"].output_fns[1].t+=k
    topo.sim["V1ComplexInh"].output_fns[1].t*=0    
    topo.sim["V1ComplexInh"].output_fns[1].t+=l


def check_activity(a,b,c,d,e,f,g,h,i,j,k,l):

    print a,b,c,d,e,f,g,h,i,j,k,l
    
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackExc1"].strength=b
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackInh"].strength=c
    topo.sim["V1Complex"].projections()["LongEE"].strength=d
    topo.sim["V1ComplexInh"].projections()["LongEI"].strength=e
    topo.sim["V1Complex"].projections()["LocalIE"].strength=f
    topo.sim["V1ComplexInh"].projections()["LocalII"].strength=g
    topo.sim["V1Complex"].projections()["V1SimpleAfferent"].strength=h
    topo.sim["V1Complex"].projections()["LocalEE"].strength=i
    topo.sim["V1ComplexInh"].projections()["LocalEI"].strength=j
    topo.sim["V1Complex"].output_fns[1].t*=0    
    topo.sim["V1Complex"].output_fns[1].t+=k
    topo.sim["V1ComplexInh"].output_fns[1].t*=0    
    topo.sim["V1ComplexInh"].output_fns[1].t+=l
    
    par = "_" + str(a)+ "_" + str(b) + "_" + str(c) + "_" + str(d)+ "_" + str(e)  + "_" + str(f) + "_" + str(g) + "_" + str(h) + "_" + str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l) +".png" 

    plot_neural_dynamics(par)

def make_full_analysis(a,b,c,d,e,f,g,h,i,j,k,l):
    import topo
    print a,b,c,d,e,f,g,h,i,j,k,l
    
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackExc1"].strength=b
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackInh"].strength=c
    topo.sim["V1Complex"].projections()["LongEE"].strength=d
    topo.sim["V1ComplexInh"].projections()["LongEI"].strength=e
    topo.sim["V1Complex"].projections()["LocalIE"].strength=f
    topo.sim["V1ComplexInh"].projections()["LocalII"].strength=g
    topo.sim["V1Complex"].projections()["V1SimpleAfferent"].strength=h
    topo.sim["V1Complex"].projections()["LocalEE"].strength=i
    topo.sim["V1ComplexInh"].projections()["LocalEI"].strength=j
    
    topo.sim["V1Complex"].output_fns[1].t*=0    
    topo.sim["V1Complex"].output_fns[1].t+=k
    topo.sim["V1ComplexInh"].output_fns[1].t*=0    
    topo.sim["V1ComplexInh"].output_fns[1].t+=l
    
    topo.sim['V1Simple'].output_fns[0].old_a*=0
    topo.sim['V1Complex'].output_fns[0].old_a*=0
    topo.sim['V1ComplexInh'].output_fns[0].old_a*=0

    from topo.analysis.featureresponses import MeasureResponseCommand, FeatureMaps, SinusoidalMeasureResponseCommand,FeatureCurveCommand
    FeatureMaps.num_orientation=16
    MeasureResponseCommand.scale=1.0
    SinusoidalMeasureResponseCommand.frequencies=[2.4]
    FeatureCurveCommand.num_orientation=16
    MeasureResponseCommand.duration=4.0
    
    V1Splastic =     topo.sim["V1Simple"].plastic
    V1Cplastic =     topo.sim["V1Complex"].plastic
    V1CInhplastic =     topo.sim["V1ComplexInh"].plastic    
    topo.sim["V1Simple"].plastic = False
    topo.sim["V1Complex"].plastic = False
    topo.sim["V1ComplexInh"].plastic = False
    wipe_out_activity()
    clear_event_queue()
    
    par = 'Analysis:' + str(a)+ "_" + str(b) + "_" + str(c) + "_" + str(d)+ "_" + str(e)  + "_" + str(f) + "_" + str(g) + "_" + str(h) + "_" + str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l)
    #d = os.path.dirname(par)
    if not os.path.exists(par):
             os.makedirs(par)
    normalize_path.prefix = par
    
    plot_neural_dynamics('neural_dynamics.png')
    
    import contrib.surround_analysis

    from topo.analysis.featureresponses import SinusoidalMeasureResponseCommand,FeatureCurveCommand
    from topo.base.projection import ProjectionSheet
    from topo.sheet.basic import GeneratorSheet
    import contrib.jacommands
    import contrib.surround_analysis
    exec "from topo.analysis.vision import analyze_complexity" in __main__.__dict__
    from topo.analysis.featureresponses import PatternPresenter            
    
    PatternPresenter.duration=4.0
    import topo.command.pylabplot
    reload(topo.command.pylabplot)                
    
    contrib.surround_analysis.run_dynamics_analysis(0.0,0.0,0.7,__main__.__dict__.get("analysis_scale",0.3))
    contrib.surround_analysis.size_tuning_analysis(0.0,0.0,1.0)
    PatternPresenter.duration=4.0
    a = topo.command.pylabplot.measure_or_tuning_fullfield.instance(sheet=topo.sim["V1Complex"])
    a.duration=4.0
    a()
    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0]",sheet=topo.sim["V1Complex"],coords=[(0,0)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0.1]",sheet=topo.sim["V1Complex"],coords=[(0.1,0.1)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,-0.1]",sheet=topo.sim["V1Complex"],coords=[(0.1,-0.1)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0.1]",sheet=topo.sim["V1Complex"],coords=[(-0.1,0.1)])()    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,-0.1]",sheet=topo.sim["V1Complex"],coords=[(-0.1,-0.1)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,0.2]",sheet=topo.sim["V1Complex"],coords=[(0.2,0.2)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.2,-0.2]",sheet=topo.sim["V1Complex"],coords=[(0.2,-0.2)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,0.2]",sheet=topo.sim["V1Complex"],coords=[(-0.2,0.2)])()    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.2,-0.2]",sheet=topo.sim["V1Complex"],coords=[(-0.2,-0.2)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.1]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.1)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.1]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.1)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.1,0]",sheet=topo.sim["V1Complex"],coords=[(-0.1,0.0)])()    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.1,0]",sheet=topo.sim["V1Complex"],coords=[(0.1,-0.0)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.3,0.3]",sheet=topo.sim["V1Complex"],coords=[(0.3,0.3)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.3,-0.3]",sheet=topo.sim["V1Complex"],coords=[(0.3,-0.3)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.3,0.3]",sheet=topo.sim["V1Complex"],coords=[(-0.3,0.3)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.3,-0.3]",sheet=topo.sim["V1Complex"],coords=[(-0.3,-0.3)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.24,0.24]",sheet=topo.sim["V1Complex"],coords=[(0.25,0.25)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.24,-0.24]",sheet=topo.sim["V1Complex"],coords=[(0.25,-0.25)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.24,0.24]",sheet=topo.sim["V1Complex"],coords=[(-0.25,0.25)])()    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.24,-0.24]",sheet=topo.sim["V1Complex"],coords=[(-0.25,-0.25)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.24]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.25)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.24]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.25)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.24,0]",sheet=topo.sim["V1Complex"],coords=[(-0.25,0.0)])()    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.24,0]",sheet=topo.sim["V1Complex"],coords=[(0.25,-0.0)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,0.3]",sheet=topo.sim["V1Complex"],coords=[(0.0,0.3)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0,-0.3]",sheet=topo.sim["V1Complex"],coords=[(0.0,-0.3)])()
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[-0.3,0]",sheet=topo.sim["V1Complex"],coords=[(-0.3,0.0)])()    
    topo.command.pylabplot.cyclic_tuning_curve.instance(x_axis="orientation",filename="ORTC[0.3,0]",sheet=topo.sim["V1Complex"],coords=[(0.3,-0.0)])()

    contrib.surround_analysis.surround_analysis().analyse([(0,0),(1.0,0.0),(0.0,1.0),(-1.0,0.0),(0.0,-1.0),(1.0,1.0),(-1.0,1.0),(1.0,-1.0),(-1.0,-1.0)])



def plot_neural_dynamics(params):

    sheet_names=["V1Complex"]

    ip = topo.sim['Retina'].input_generator
    topo.sim['Retina'].set_input_generator(SineGratingDiskTemp(orientation=0.0,phase=0.0,size=10,scale=1.0,x=0.0,y=0.0,frequency=2.4))

    from topo.pattern.basic import OrientationContrast
    from topo.command.basic import pattern_present
    from topo.base.functionfamily import PatternDrivenAnalysis
    from topo.pattern.basic import OrientationContrast
    from topo.analysis.featureresponses import PatternPresenter
    from topo.base.sheet import Sheet
    import pylab

    topo.sim['V1Simple'].output_fns[0].old_a*=0
    topo.sim['V1Complex'].output_fns[0].old_a*=0
    topo.sim['V1ComplexInh'].output_fns[0].old_a*=0

    V1Splastic =     topo.sim["V1Simple"].plastic
    V1Cplastic =     topo.sim["V1Complex"].plastic
    V1CInhplastic =     topo.sim["V1ComplexInh"].plastic    
    topo.sim["V1Simple"].plastic = False
    topo.sim["V1Complex"].plastic = False
    topo.sim["V1ComplexInh"].plastic = False

    prefix="/home/antolikjan/topographica/ActivityExploration/"
    
    topo.sim.state_push()
    
    from topo.command.basic import pattern_present
    from topo.base.functionfamily import PatternDrivenAnalysis
    from topo.pattern.basic import OrientationContrast
    from topo.analysis.featureresponses import PatternPresenter
    from topo.base.sheet import Sheet
    
    data={}
    
    for key in sheet_names:
        data[key] = {}
        for i in topo.sim[key].projections().keys():
            data[key][i]=[]
        data[key]["act"]=[]


    for i in xrange(0,100):
	topo.sim.run(0.05)        
        for key in sheet_names:
            for i in topo.sim[key].projections().keys():
                data[key][i].append(topo.sim[key].projections()[i].activity.copy())
            data[key]["act"].append(topo.sim[key].activity.copy())
    
    acts = topo.sim["V1Simple"].activity.copy()      
    actc = topo.sim["V1Complex"].activity.copy()      

    topo.sim.state_pop()        

    m = numpy.argmax(data["V1Complex"]["act"][-1])
    (X,Y) = numpy.unravel_index(m, data["V1Complex"]["act"][-1].shape)

    print X,Y

    pylab.figure(figsize=(20,5))
    pylab.subplot(2,3,1)
    pylab.title(prefix+sheet_names[0]+" [" + str(X) + "," +str(Y) + "]")
    for projname in data[sheet_names[0]].keys():
        a = []
        for act in data[sheet_names[0]][projname]:
            a.append(act[X,Y])
        pylab.plot(a,label=projname)
    #pylab.legend(loc='upper left')
    pylab.subplot(2,3,2)
    pylab.imshow(acts)
    pylab.colorbar()
    
    pylab.subplot(2,3,3)
    pylab.imshow(actc)
    pylab.colorbar()

    (xx,yy) = topo.sim["V1Complex"].matrixidx2sheet(X,Y)
    # now lets collect the size tuning 
    step_size=0.2
    stc_lc = []
    for i in xrange(0,20):
        topo.sim['V1Simple'].output_fns[0].old_a*=0
        topo.sim['V1Complex'].output_fns[0].old_a*=0
        topo.sim['V1ComplexInh'].output_fns[0].old_a*=0
        wipe_out_activity()
        clear_event_queue()

	topo.sim['Retina'].set_input_generator(SineGratingDiskTemp(orientation=0.0,phase=0.0,size=i*step_size,scale=0.3,x=xx,y=yy,frequency=2.4))
	topo.sim.state_push()     
	topo.sim.run(4.0)           
        stc_lc.append(topo.sim["V1Complex"].activity[X,Y].copy())
        topo.sim.state_pop()        


    stc_hc = []
    for i in xrange(0,20):
        topo.sim['V1Simple'].output_fns[0].old_a*=0
        topo.sim['V1Complex'].output_fns[0].old_a*=0
        topo.sim['V1ComplexInh'].output_fns[0].old_a*=0
        wipe_out_activity()
	clear_event_queue()
    	topo.sim['Retina'].set_input_generator(SineGratingDiskTemp(orientation=0.0,phase=0.0,size=i*step_size,scale=1.0,x=xx,y=yy,frequency=2.4))
	topo.sim.state_push()     
	topo.sim.run(4.0)           
        stc_hc.append(topo.sim["V1Complex"].activity[X,Y].copy())
        topo.sim.state_pop()        
        
    pylab.subplot(2,3,4)
    pylab.plot(stc_lc,label='30%')
    pylab.plot(stc_hc,label='100%')
    pylab.legend()

    topo.sim["V1Simple"].plastic = V1Splastic
    topo.sim["V1Complex"].plastic = V1Cplastic
    topo.sim["V1ComplexInh"].plastic = V1CInhplastic
    wipe_out_activity()
    clear_event_queue()
    
    topo.sim['Retina'].set_input_generator(ip)
    pylab.savefig(prefix+ sheet_names[0] + params);    





#contrib.jacommands.run_combinations(check_activity,[[0],[0.1],[-2.5],[0.1],[4.0,5.0,6.0],[-1.0,-1.1],[-0.9,-0.8],[3.0],[1.7],[2.2],[0.1],[0.2,0.3]])

#make_full_analysis(0,0.1,-2.5,0.1,6.0,-1.1,-0.8,3.0,1.7,2.2,0.1,0.3)
#make_full_analysis(0,0.1,-2.5,0.4,0.1,-8.0,-1.2,1.5,0.7,0.15,0.1,0.3)

contrib.jacommands.run_combinations(check_activity,[[0],[0.1],[-2.5,-2.0,-3.0],[0.4,0.8],[0.1,0.2],[-8.0,-5.0,-6.0],[-1.2],[1.0],[0.6],[0.1],[0.05,0.1],[0.1,0.2]])
#contrib.jacommands.run_combinations(check_activity,[[0],[0.05,0.1,0.15],[-1.0,-2.0,-3.0],[0.4],[0.1],[-5.7],[-1.7],[0.5],[1.15,1.0,1.3],[0.2,0.15,0.25]])

