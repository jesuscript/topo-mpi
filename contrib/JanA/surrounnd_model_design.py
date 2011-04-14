from topo.analysis.featureresponses import MeasureResponseCommand, FeatureMaps, SinusoidalMeasureResponseCommand, FeatureCurveCommand
FeatureMaps.num_orientation=16
MeasureResponseCommand.scale=1.0
MeasureResponseCommand.duration=4.0
SinusoidalMeasureResponseCommand.frequencies=[2.4]
FeatureCurveCommand.num_orientation=16
FeatureCurveCommand.curve_parameters=[{"contrast":15},{"contrast":50},{"contrast":90}]

def check_activity(a,b,c,d,e,f,g,h,i,j):
    
    topo.sim['V1Simple'].output_fns[0].old_a*=0
    topo.sim['V1Complex'].output_fns[0].old_a*=0
    topo.sim['V1ComplexInh'].output_fns[0].old_a*=0

    print a,b,c,d,e,f,g,h,i,j
    topo.pattern.random.seed(13)
    import pylab
    prefix="./ActivityExploration/"
    
    #pylab.figure(1)

    V1Splastic =     topo.sim["V1Simple"].plastic
    V1Cplastic =     topo.sim["V1Complex"].plastic
    V1CInhplastic =     topo.sim["V1ComplexInh"].plastic    
    topo.sim["V1Simple"].plastic = False
    topo.sim["V1Complex"].plastic = False
    topo.sim["V1ComplexInh"].plastic = False
    
    topo.sim.state_push()
   
    #topo.sim["V1Simple"].in_connections[0].strength=a
    #topo.sim["V1Simple"].in_connections[0].strength=a
    
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackExc1"].strength=b
    topo.sim["V1Simple"].projections()["V1SimpleFeedbackInh"].strength=c
    topo.sim["V1Complex"].projections()["LongEE"].strength=d
    topo.sim["V1ComplexInh"].projections()["LongEI"].strength=e
    topo.sim["V1Complex"].projections()["LocalIE"].strength=f
    topo.sim["V1ComplexInh"].projections()["LocalII"].strength=g
    topo.sim["V1Complex"].projections()["V1SimpleAfferent"].strength=h
    topo.sim["V1Complex"].projections()["LocalEE"].strength=i
    topo.sim["V1ComplexInh"].projections()["LocalEI"].strength=j

    par = "_" + str(a)+ "_" + str(b) + "_" + str(c) + "_" + str(d)+ "_" + str(e)  + "_" + str(f) + "_" + str(g) + "_" + str(h) + "_" + str(i) + "_" + str(j) + ".png"    
    
    try:
        
        evolution=[]
	evolution_simple=[]
        for k in xrange(0,100):
	    topo.sim.run(0.05)
	    evolution.append(topo.sim["V1Complex"].activity.copy())
	    evolution_simple.append(topo.sim["V1Simple"].activity.copy())
	    
	    if k==0:
		activity = topo.sim["V1Complex"].activity.copy()
	    
	    if k==70 or k==80 or k==90 or k==100:
		activity += topo.sim["V1Complex"].activity
		
	m = numpy.argmax(topo.sim["V1Complex"].activity)
	(X,Y) = numpy.unravel_index(m, topo.sim["V1Complex"].activity.shape)
	
        evolution1=[]
	evolution_simple1=[]

	for s in evolution:
	    evolution1.append(s.ravel()[m])

	for s in evolution_simple:
	    evolution_simple1.append(s.ravel()[m])


		
        if  topo.sim["V1Complex"].activity[36][36] < 1000000000000:
            print 'OK'
            pylab.figure()
	    pylab.subplot(2,2,1)
    	    pylab.imshow(topo.sim["V1Complex"].activity)
            pylab.subplot(2,2,2)
	    pylab.plot(evolution1)
	    pylab.subplot(2,2,3)
    	    #pylab.imshow(topo.sim["FakeRetina"].activity)
    	    pylab.plot(evolution_simple1)
	    pylab.subplot(2,2,4)
    	    pylab.imshow(topo.sim["V1Simple"].activity)

        evolution=[]
	evolution_simple=[]

        pylab.savefig(prefix+ "Activity:" + par );


    except FloatingPointError:
        print "Error"
        pass
    except AttributeError:
	print 'AttributeError:'
        pass
    
    topo.sim.state_pop()
    
    topo.sim["V1Simple"].plastic = V1Splastic
    topo.sim["V1Complex"].plastic = V1Cplastic
    topo.sim["V1ComplexInh"].plastic = V1CInhplastic
    wipe_out_activity()
    clear_event_queue()

    plot_neural_dynamics(par,0.0,0.0)
    
from topo.command.basic import wipe_out_activity, clear_event_queue
topo.sim.run(1.0)
wipe_out_activity()
clear_event_queue()

from topo.pattern.basic import SineGrating, Disk
class SineGratingDiskTemp(SineGrating):
      mask_shape = param.Parameter(default=Disk(smoothing=0,size=1.0))

def size_tuning_analysis(x,y,scale,params):
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

    prefix="./ActivityExploration/"
    
    topo.sim.state_push()

  
    (xx,yy) = topo.sim["V1Complex"].sheet2matrixidx(x,y)
    
    orr=0 
    phase =0 
    
    activities_s = []
    activities_c = []
    activities_ci = []
    
    for i in xrange(0,40):
	pg = SineGratingDiskTemp(orientation=orr,phase=phase,size=(12.0/float(i+1)),scale=1.0,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))
	
	pp = PatternPresenter(pattern_generator=pg,duration=4.0,contrast_parameter="weber_contrast")
	    
	for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
	topo.sim.state_push()
	for f in PatternDrivenAnalysis.pre_presentation_hooks: f()

	pp({},{})
	topo.guimain.refresh_activity_windows()
	
	activities_c.append(topo.sim["V1Complex"].activity.copy())
	activities_ci.append(topo.sim["V1ComplexInh"].activity.copy())
	activities_s.append(topo.sim["V1Simple"].activity.copy())

	for f in PatternDrivenAnalysis.post_presentation_hooks: f()
	topo.sim.state_pop()
	for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
    
    
    a = []
    b = []
    c = []

#    pylab.figure()
#    pylab.subplot(6,6,1)
    for i in xrange(0,40):	
#    	pylab.subplot(7,7,i+1)
#        pylab.imshow(activities_c[i],vmin=0.0,vmax=2.0,interpolation='nearest')
#        pylab.xticks([], [])
#	pylab.yticks([], [])
#	pylab.xlabel(str(12.0-i*0.3),fontsize=8)
	#pylab.colorbar(shrink=0.1)
	a.append(activities_c[i][xx][yy])
	c.append(activities_ci[i][xx][yy])
	b.append(12.0-i*0.3)

#    pylab.figure()
#    pylab.subplot(6,6,1)
#    for i in xrange(0,40):	
#	pylab.subplot(7,7,i+1)
#        pylab.imshow(activities_s[i],vmin=0.0,vmax=2.0,interpolation='nearest')
#        pylab.xticks([], [])
#	pylab.yticks([], [])
#	pylab.xlabel(str(12.0-i*0.3),fontsize=8)

    pylab.figure()
    pylab.subplot(2,1,1)
    pylab.plot(b,a)
    pylab.subplot(2,1,2)
    pylab.plot(b,c)

    pylab.savefig(prefix+ "STC" + params);

    topo.sim["V1Simple"].plastic = V1Splastic
    topo.sim["V1Complex"].plastic = V1Cplastic
    topo.sim["V1ComplexInh"].plastic = V1CInhplastic
    wipe_out_activity()
    clear_event_queue()


def plot_neural_dynamics(params,x,y):

    sheet_names=["V1Complex"]
    neurons=[("V1Complex",(x,y))]

    ip = topo.sim['FakeRetina'].input_generator
    topo.sim['FakeRetina'].set_input_generator(SineGratingDiskTemp(orientation=0.0,phase=0.0,size=10,scale=0.05,x=0.0,y=0.0,frequency=2.4))

    	
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

    prefix="./ActivityExploration/"
    
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
    
    act = topo.sim["V1Complex"].activity.copy()      

    topo.sim.state_pop()        

    for n in neurons:
       (sheetname, (x,y)) = n
       #(xx,yy) = (x,y)
       #(xx,yy) = topo.sim[sheetname].sheet2matrixidx(x,y)

       m = numpy.argmax(data["V1Complex"]["act"][-1])
       (X,Y) = numpy.unravel_index(m, data["V1Complex"]["act"][-1].shape)


       pylab.figure()
       pylab.subplot(1,2,1)
       pylab.title(prefix+sheetname+" [" + str(X) + "," +str(Y) + "]")
       for projname in data[sheetname].keys():
           a = []
           for act in data[sheetname][projname]:
               a.append(act[X,Y])
           pylab.plot(a,label=projname)
       pylab.legend(loc='upper left')
       pylab.subplot(1,2,2)
       pylab.imshow(act)
       pylab.colorbar()
       pylab.savefig(prefix+ sheetname + params);    

    topo.sim["V1Simple"].plastic = V1Splastic
    topo.sim["V1Complex"].plastic = V1Cplastic
    topo.sim["V1ComplexInh"].plastic = V1CInhplastic
    wipe_out_activity()
    clear_event_queue()
    topo.sim['FakeRetina'].set_input_generator(ip)

#contrib.jacommands.run_combinations(check_activity,[[0],[0.15],[-3.1],[0.4],[0.1],[-5.2],[-1.9],[0.5],[1.3],[0.1]])


#contrib.jacommands.run_combinations(check_activity,[[0],[0.15],[-3.1],[0.4],[0.1],[-6.2,-6.7,-6.9],[-1.9,-1.5,-1.7],[0.5],[1.1,1.0,0.9],[0.1,0.2,0.15]])
#contrib.jacommands.run_combinations(check_activity,[[0],[0.05,0.1,0.15],[-1.0,-2.0,-3.0],[0.4],[0.1],[-5.7],[-1.7],[0.5],[1.15,1.0,1.3],[0.2,0.15,0.25]])

