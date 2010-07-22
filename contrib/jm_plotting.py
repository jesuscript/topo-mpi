from math import pi,sin,cos,sqrt, ceil

import numpy
from numpy import array, zeros

try:
    import pylab
except ImportError:
    print "Warning: Could not import matplotlib; pylab plots will not work."

import topo
import __main__

from topo.base.cf import CFSheet
from topo.plotting.plotgroup import create_plotgroup, plotgroups
from topo.command.pylabplots import PylabPlotCommand
from topo.base.sheetview import SheetView
from topo.base.projection import ProjectionSheet
from param.parameterized import ParamOverrides
from topo.misc.util import frange

from contrib.jm_analysis import *

#
#  batch function
#
def complex_analysis_function():
    """
    Analysis command for run_batch; Saves maps of orientation preference and analyzes complex cells
    using Modulation Ratio, Circular Variance and Orientation Bandwidth. Additionally, it saves projection
    plots.
    """
    import topo
    from topo.command.analysis import save_plotgroup
    from topo.base.projection import ProjectionSheet

    save_plotgroup("Orientation Preference, Modulation Ratio, Circular Variance and Orientation Bandwidth")
    save_plotgroup("Activity")

    # Plot all projections for all measured_sheets
    measured_sheets = [s for s in topo.sim.objects(ProjectionSheet).values()
                       if hasattr(s,'measure_maps') and s.measure_maps]
    for s in measured_sheets:
        for p in s.projections().values():
            save_plotgroup("Projection",projection=p)

#
#   General plot functions
#
class plot_correlation_from_view(PylabPlotCommand):
    """
    This function plots dependency (which is in fact a 2D histogram) of two views for
    each sheet in sheets_to_plot and for sum of all those sheets.

    Sheets' views are specified with x_view_name and  y_view_name.
    x_axis (and y_axis) is a triple defining bounds for each plot and the precision step.
    x_axis = (x_min,x_max,dx)
    """
    def __call__(self, x_view_name, x_label, x_axis, y_view_name, y_label, y_axis, sheets_to_plot=[], **params):
        p=ParamOverrides(self,params)
           

        if len(sheets_to_plot) > 8:
            self.warning( "Too many sheets to plot. Plotting first 8 of them." )
            sheets_to_plot=sheets_to_plot[:8]

        subplot_base_number=int(100*(ceil(float(len(sheets_to_plot)+1)/2))+20)
        num_plotted_sheets=0

        # bounds and step for the bin precision
        x_min,x_max,dx=x_axis
        y_min,y_max,dy=y_axis
        # number of steps
        XXX=ceil((x_max-x_min)/dx)
        YYY=ceil((y_max-y_min)/dy)
        
        # 2D field of bins (actually a 2D histogram...) for a corr accross all sheets
        mesh_z_total=zeros((ceil((y_max-y_min)/dy),ceil((x_max-x_min)/dx)))

        pylab.figure()

        for name in sheets_to_plot:
            # get the sheet object
            try:
                sheet=topo.sim.objects()[name]
            except KeyError:
                self.warning( "Name '"+name+"' is not present in Topographica objects. Skipping.")
                continue
            # get the y-axes view
            try:
                y_view=sheet.sheet_views[y_view_name].view()[0]
            except KeyError:
                self.warning( "View "+y_view_name+" is not present in '"+name+"' views. Skipping.")
                continue
            # get the x-axes view
            try:
                x_view=sheet.sheet_views[x_view_name].view()[0]
            except KeyError:
                self.warning( "View "+x_view_name+" is not present in '"+name+"' views. Skipping.")
                continue
            num_plotted_sheets += 1
            
            a_shape, b_shape = y_view.shape
            # initialize the only-for-this-sheet 2D hist.
            mesh_z_local=zeros((ceil((y_max-y_min)/dy),ceil((x_max-x_min)/dx)))

            # This would be weird....
            if y_view.shape != x_view.shape:
                self.warning( "Views '"+x_view_name+"' and '"+y_view_name+"' in '" + name + 
                              "' Don't have the same shapes. This is weird. Skipping.")
                continue
            # get list of paired values (view 1, view 2)    
            list_tuple=[(x_view[a][b], y_view[a][b]) for a in xrange(a_shape) for b in xrange(b_shape)]

            # throw each value into correct bin in our 2D histogram mesh_z
            for a,b in list_tuple:
                if b > y_max or a > x_max:
                    self.warning("(%f,%f) exceeds bounds (%f,%f), skipping."%(a,b,x_max,y_max))
                    continue
                AAA=ceil((a-x_min)/dx)
                BBB=ceil((b-y_min)/dy)
                if AAA > XXX or BBB > YYY:
                    self.warning("(%d,%d) exceeds bounds (%d,%d), skipping."%(AAA,BBB,XXX,YYY))
                    break
                if AAA <=0 or BBB <= 0:
                    self.warning("(%d,%d) zero for (a,b) (%f,%f), skipping."%(AAA,BBB,a,b))
                    break
                mesh_z_local[ceil((b-y_min)/dy)-1][ceil((a-x_min)/dx)-1]+=1
            # add the local corr to the total corr
            mesh_z_total += mesh_z_local
            # plot it
            pylab.subplot(subplot_base_number+num_plotted_sheets)
            pylab.title(name,fontsize=10)
            pylab.ylabel(y_label)
            pylab.xlabel(x_label)
            pylab.imshow(mesh_z_local, interpolation='bilinear', origin='lower', extent=[x_min,x_max,y_min,y_max])

        # if there is only one sheet, sum of all sheets is redundant
        if num_plotted_sheets > 1:    
            num_plotted_sheets += 1
            pylab.subplot(subplot_base_number+num_plotted_sheets)
            pylab.title('Sum of All Sheets',fontsize=10)
            pylab.ylabel(y_label)
            pylab.xlabel(x_label)
            pylab.imshow(mesh_z_total, interpolation='bilinear', origin='lower', extent=[x_min,x_max,y_min,y_max])
        self._generate_figure(p)

class plot_hist_from_view(PylabPlotCommand):
    """
    This function plots histogram and quantiles for a specified view and sheets.
    Furthermore it computes histogram of sum of views from the sheets.

    View name is specified with view_name.
    axis is a triple defining bounds for each plot and the size of a histogram bin.
    axis = (x_min,x_max,dx)
    """
    
    # I guess this is pretty nice and should be used for some other plots (like modulation ratio)

    def __call__(self, view_name, axis, xlabel='', sheets_to_plot=[], quantiles=[0.25,0.5,0.75],**params):
        """
        Parameter sheets_to_plot can be any list of valid sheet names.
        They must have view_name view present to obtain valid plot.
        """
        p=ParamOverrides(self,params)
        if len(sheets_to_plot) > 8:
            self.warning( "Too many sheets to plot. Plotting first 8 of them." )
            sheets_to_plot=sheets_to_plot[:8]
        
        # list of values across all cortical sheets
        list_total=[]
        subplot_base_number=100*(len(sheets_to_plot)+1)+10
        num_plotted_sheets=0

        x_min,x_max,dx=axis

        pylab.figure()

        for name in sheets_to_plot:
            # get the sheet object
            try:
                sheet=topo.sim.objects()[name]
            except KeyError:
                self.warning( "Name '"+name+"' is not present in Topographica objects. Skipping.")
                continue
            # get the histogrammed view
            try:
                view=sheet.sheet_views[view_name].view()[0]
            except KeyError:
                self.warning( "View '"+view_name+"' is not present in '"+name+"' views. Skipping.")
                continue
            num_plotted_sheets += 1
            
            # get linear list of values
            list_local = [ element for row in view for element in row ]
            list_total += list_local

            # plot it
            pylab.subplot(subplot_base_number+num_plotted_sheets)
            pylab.title(name,fontsize=10)
            pylab.ylabel('Number Of Cells')
            hist = pylab.hist(list_local,frange(x_min,x_max,dx))          #,inclusive=True))
            # Compute 1000 ceil of bin with highest number of cells
            # the plot is nicer this way
            upper_bound = ceil(float(hist[0].max())/500)*500
            pylab.axis([x_min,x_max,0,upper_bound])

            # plot the quantiles right below
            list_local.sort()
            list_length=len(list_local)
            for quantile in quantiles:
                quantile_value=list_local[int(list_length*quantile)]
                pylab.plot([quantile_value]*2,[upper_bound,0],'r' if quantile == 0.5 else 'g')

    # if there would be only one sheet, sum of all sheets would be redundant
        if num_plotted_sheets > 1:    
            num_plotted_sheets += 1
            pylab.subplot(subplot_base_number + num_plotted_sheets)
            pylab.title('Sum of All sheets',fontsize=10)
            pylab.ylabel('Number Of Cells')
            pylab.xlabel(xlabel)
            hist = pylab.hist(list_total,frange(x_min,x_max,dx))          # ,inclusive=True))
            # Compute 1000 ceil of bin with highest number of cells
            # the plot is nicer this way
            upper_bound = ceil(float(hist[0].max())/1000)*1000
            pylab.axis([x_min,x_max,0,upper_bound])
                
            list_total.sort()
            list_length=len(list_total)
            for quantile in quantiles:
                quantile_value=list_total[int(list_length*quantile)]
                pylab.plot([quantile_value]*2,[upper_bound,0],'r' if quantile == 0.5 else 'g')

        self._generate_figure(p)

# substitutes phase_preference_scatter_plot from topo.analysis.vision
class scatter_plot_phase_preference_simple_cell(PylabPlotCommand):
    """
    This function plots phase preference scatter plot for pairs of simple cells present in sheet specified with sheet_name.
    Note that Modulation ratio and Phase preference must be in available sheet views.
    """

    def __call__(self, sheet_name, num_of_pairs=66, diameter=0.39, **params):
        p=ParamOverrides(self,params)

        from topo import numbergen

        # init random generator
        r =numbergen.UniformRandom(seed=1023)
        # get the sheet object
        try:
            sheet=topo.sim.objects()[sheet_name]
        except KeyError:
            self.warning( "Name '"+sheet_name+"' is not present in Topographica objects. Skipping.")
            return
        # get the phase pref view
        try:
            phase_preference = sheet.sheet_views['PhasePreference'].view()[0]
        except KeyError:
            self.warning( "View 'PhasePreference' is not present in '"+sheet_name+"' views. Skipping.")
            return
        # get the mr view
        try:
            modulation_ratio = sheet.sheet_views['ModulationRatio'].view()[0]
        except KeyError:
            self.warning( "View 'ModulationRatio' is not present in '"+sheet_name+"' views. Skipping.")
            return

        offset_magnitude = 0.03
        datax = []
        datay = []
        # while we do not have enough pairs of simple cells
        while len(datax) < num_of_pairs:
            x,y,xx,yy=0,0,0,0
            count1=0
            # while not enough simple cells
            # (not a joke, see a break inside the loop)
            while True:
                count1 += 1
                # get the first cell X
                x_s = max(min((r() - 0.5)*2*diameter,diameter),-diameter)
                y_s = max(min((r() - 0.5)*2*diameter,diameter),-diameter)
                (x,y) = topo.sim[sheet_name].sheet2matrixidx(x_s,y_s)
                # if the cell is simple
                if modulation_ratio[x][y] >= 1 :
                    count2=0
                    # while not found a simple cell from the X's proximity
                    while True:
                        count2 += 1
                        rand = r()
                        # find a cell in X's proximity
                        xoff = sin(rand*2*pi)*offset_magnitude
                        yoff = cos(rand*2*pi)*offset_magnitude
                        xx_s = max(min(x_s+xoff,diameter),-diameter)
                        yy_s = max(min(y_s+yoff,diameter),-diameter)
                        (xx,yy) = topo.sim[sheet_name].sheet2matrixidx(xx_s,yy_s)
                        # is it simple?
                        if modulation_ratio[xx][yy] >= 1 and not (xx == x and yy == y) :
                            break
                        # if tried too many times, there probably is not enough simple cells  around...
                        if count2 >= 1000:
                            self.warning("Not enough simple cells in sheet '"+sheet_name+"', giving up.")
                            return
                    break

                # if tried too many times, there probably is not enough simple cells  around...
                if count1 >= 1000:
                    self.warning("Not enough simple cells in sheet '"+sheet_name+"', giving up.")
                    return
            # extend the list of proximate cells
            datax += [phase_preference[x,y]]
            datay += [phase_preference[xx,yy]]

        # normalize the phase preferences
        for i in xrange(len(datax)):
            datax[i] = datax[i] * 360
            datay[i] = datay[i] * 360
            if(datay[i] > datax[i] + 180): datay[i]=  datay[i]- 360
            if((datax[i] > 180) & (datay[i]> 180)): datax[i] = datax[i] - 360; datay[i] = datay[i] - 360
            if((datax[i] > 180) & (datay[i] < (datax[i]-180))): datax[i] = datax[i] - 360; #datay[i] = datay[i] - 360

        # plot the hell out of it!!    
        f = pylab.figure()
        ax = f.add_subplot(111, aspect='equal')
        pylab.plot(datax,datay,'ro')
        pylab.title('Phase preference scatter plot, sheet '+sheet_name,fontsize=10)
        pylab.plot([0,360],[-180,180])
        pylab.plot([-180,180],[0,360])
        pylab.plot([-180,-180],[360,360])
        ax.axis([-180,360,-180,360])
        pylab.xticks([-180,0,180,360], [-180,0,180,360])
        pylab.yticks([-180,0,180,360], [-180,0,180,360])
        pylab.grid()

        self._generate_figure(p)
#
#   Analyze functions, compute the statistic from full_matrix, plot its hist
#
def analyze_circular_variance(full_matrix, sheets_to_plot=[], filename=None):
    """
    Computes the circular variance for each sheet in a sheets_to_plot and save appropriate views.

    Uses full_matrix data obtained from measure_or_pref().
    """
    import topo
    measured_sheets = [s for s in topo.sim.objects(CFSheet).values()
                          if hasattr(s,'measure_maps') and s.measure_maps]

    for sheet in measured_sheets:   
        # compute the cv and save the view
        cv = circular_variance(full_matrix[sheet])
        sheet.sheet_views['CircularVariance']=SheetView((cv,sheet.bounds), sheet.name , sheet.precedence, topo.sim.time(),sheet.row_precedence)
    
    if len(sheets_to_plot) > 0:
        # plot hist for sheets_to_plot
        plot_hist_from_view('CircularVariance', (0,1,0.1), xlabel='Circular Variance',
                            sheets_to_plot=sheets_to_plot, filename='Hist'+filename if filename != None else None)

# this method substitutes topo.analysis.vision analyze_complexity
def analyze_modulation_ratio(full_matrix, sheets_to_plot=[], sheets_to_scatter_plot=[], filename=None):
    """
    Computes modulation ratio for each neuron, to distinguish complex from simple cells.

    Uses full_matrix data obtained from measure_or_pref().

    Plots phase preference scatter plot for sheets specified in sheets_to_scatter_plot
    """
    import topo
    measured_sheets = [s for s in topo.sim.objects(CFSheet).values()
                          if hasattr(s,'measure_maps') and s.measure_maps]

    for sheet in measured_sheets:   
        from topo.analysis.vision import complexity
        mr = array(complexity(full_matrix[sheet]))
        sheet.sheet_views['ModulationRatio']=SheetView((mr,sheet.bounds), sheet.name, sheet.precedence, topo.sim.time(), sheet.row_precedence)

    if len(sheets_to_plot) > 0:
        plot_hist_from_view('ModulationRatio', (0,2,0.1), xlabel='Modulation Ratio',
                            sheets_to_plot=sheets_to_plot, filename='Hist'+filename if filename != None else None)
    if len(sheets_to_scatter_plot) > 0:
        for sheet in sheets_to_scatter_plot:
            scatter_plot_phase_preference_simple_cell(sheet,diameter=0.24999, filename='Simple_cells_scatter_plot_'+sheet if filename != None else None)

def analyze_orientation_bandwidth(full_matrix, sheets_to_plot=[], height=1/sqrt(2), filename=None):
    """
    Computes orientation bandwidth for each neuron.

    Uses full_matrix data obtained from measure_or_pref().
    """
    measured_sheets = [s for s in topo.sim.objects(CFSheet).values()
                          if hasattr(s,'measure_maps') and s.measure_maps]

    for sheet in measured_sheets:   
        mr = orientation_bandwidth(full_matrix[sheet], height)

        sheet.sheet_views['OrientationBandwidth']=SheetView((mr,sheet.bounds), sheet.name, sheet.precedence, topo.sim.time(), sheet.row_precedence)

    if len(sheets_to_plot) > 0:
        plot_hist_from_view('OrientationBandwidth', (0,pi,0.2), xlabel='Orientation Bandwidth',
                            sheets_to_plot=sheets_to_plot, filename='Hist'+filename if filename != None else None)

def analyze_mean_activity(full_matrix, sheets_to_plot=[], filename=None):
    """
    Computes mean activity for each neuron.

    Uses full_matrix data obtained from measure_or_pref().
    """
    measured_sheets = [s for s in topo.sim.objects(CFSheet).values()
                          if hasattr(s,'measure_maps') and s.measure_maps]
    min, max=[],[]
    for sheet in measured_sheets:   
        ma = mean_activity(full_matrix[sheet])
        min += [ma.min()]
        max += [ma.max()]
        sheet.sheet_views['MeanActivity']=SheetView((ma,sheet.bounds), sheet.name, sheet.precedence, topo.sim.time(), sheet.row_precedence)
    
    minn=array(min).min()
    maxn=array(max).max()

    if len(sheets_to_plot) > 0:
        plot_hist_from_view('MeanActivity', (minn,maxn,abs(maxn-minn)/20), xlabel='MeanActivity',
                            sheets_to_plot=sheets_to_plot, filename='Hist'+filename if filename != None else None)

        plot_correlation_from_view('ModulationRatio', 'Modulation Ratio', (0,2,0.04),
                                   'MeanActivity', 'Mean Activity', (minn,maxn,abs(maxn-minn)/20),

                                   sheets_to_plot=sheets_to_plot,filename='Correlation_ModulationRatio_MeanActivity'if filename else None)
        plot_correlation_from_view('OrientationBandwidth','Orientation Bandwidth [rad]', (0,3.2,0.04),
                                       'MeanActivity', 'Mean Activity', (minn,maxn,abs(maxn-minn)/20),
                                       sheets_to_plot=sheets_to_plot,filename='Correlation_OrientationBandwidth_MeanActivity'if filename else None)
        plot_correlation_from_view('CircularVariance', 'Circular Variance', (0,1,0.02),
                                       'MeanActivity', 'Mean Activity', (minn,maxn,abs(maxn-minn)/20),
                                       sheets_to_plot=sheets_to_plot,filename='Correlation_CircularVariance_MeanActivity'if filename else None)
#
#   Macro for plotgroup
#


def measure_and_analyze_mr_cv_ob(plot_into_file=True):
    """
    Macro for measuring orientation preference and then analyzing its modulation ratio, circular variance and orientation bandwidth.
    It is recommended to set high number of orientations when measuring responses.
    E.g. 
      from topo.analysis.featureresponses import SinusoidalMeasureResponseCommand
      SinusoidalMeasureResponseCommand.num_orientation=12
    """
    from topo.command.analysis import measure_or_pref

    measured_sheets = [s for s in topo.sim.objects(CFSheet).values()
                          if hasattr(s,'measure_maps') and s.measure_maps]

    sheets= [ sheet.name for sheet in measured_sheets ]

    # collect the responses
    fm = measure_or_pref()
    # analyze each property
    #print "Modulation Ratio"
    analyze_modulation_ratio(fm, sheets_to_plot=sheets, sheets_to_scatter_plot=sheets, filename="ModulationRatio" if plot_into_file else None)
    #print "Circular Variance"
    analyze_circular_variance(fm, sheets_to_plot=sheets, filename="CircularVariance" if plot_into_file else None)
    #print "Orientation Bandwidth"
    analyze_orientation_bandwidth(fm, sheets_to_plot=sheets, filename="Bandwidth" if plot_into_file else None)

    # plot joint statistics, such as corelations, ...
    #print "Plotting correlations"
    plot_correlation_from_view('ModulationRatio', 'Modulation Ratio', (0,2,0.04),
                               'CircularVariance', 'Circular Variance', (0,1,0.02),
                               sheets_to_plot=sheets,filename='Correlation_ModulationRatio_CircularVariance'if plot_into_file else None)
    plot_correlation_from_view('ModulationRatio', 'Modulation Ratio', (0,2,0.04),
                               'OrientationBandwidth', 'Orientation Bandwidth [rad]', (0,3.2,0.04),
                               sheets_to_plot=sheets,filename='Correlation_ModulationRatio_OrientationBandwith'if plot_into_file else None)
    plot_correlation_from_view('OrientationBandwidth','Orientation Bandwidth [rad]', (0,3.2,0.04),
                               'CircularVariance', 'Circular Variance', (0,1,0.02),
                               sheets_to_plot=sheets,filename='Correlation_OrientationBandwidth_CircularVariance'if plot_into_file else None)

pg= create_plotgroup(name='Orientation Preference, Modulation Ratio, Circular Variance and Orientation Bandwidth',category="Preference Maps",
             doc='Measure preference for sine grating orientation.',
             pre_plot_hooks=[measure_and_analyze_mr_cv_ob])
pg.add_plot('Orientation Preference',[('Hue','OrientationPreference')])
pg.add_plot('Orientation Preference&Selectivity',[('Hue','OrientationPreference'),
                           ('Confidence','OrientationSelectivity')])
pg.add_plot('Orientation Selectivity',[('Strength','OrientationSelectivity')])
pg.add_plot('Modulation Ratio',[('Strength','ModulationRatio')])
pg.add_plot('Phase Preference',[('Hue','PhasePreference')])
pg.add_plot('Circular Variance',[('Strength','CircularVariance')])
pg.add_plot('Orientation Bandwidth',[('Strength','OrientationBandwidth')])
pg.add_static_image('Color Key','topo/command/or_key_white_vert_small.png')

