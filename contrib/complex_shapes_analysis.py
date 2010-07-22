"""
Complex shapes analysis and measuring of responses for Hegde and Van Essen stimuli sets.
Contains measuring of best subclasses, modulation by complex shapes etc.

$Id$
"""
__version__ = "$Revision$"

import operator
import random

import numpy
try:
    import matplotlib
    from matplotlib.ticker import NullFormatter
    import pylab
except ImportError:
    print "Warning: Could not import matplotlib; pylab plots will not work."

import topo
from topo.plotting.plotgroup import create_plotgroup, plotgroups
from topo.command.pylabplots import PylabPlotCommand
from param.parameterized import ParamOverrides
from topo.base.boundingregion import BoundingBox

def measure_responses(stimuli_subclasses, sheet_to_measure='V1', neurons_to_measure=360, rf_size=0.5, rf_jitter = 0.125):
    """
    Measuring of responses for given set of stimuli, which are expected to be grouped by subclasses
    (for example contrib.hegdeessen.all_stimuli_subclasses). Receptive field size is set equally for
    all measured neurons. These are selected by random from given sheet. Each stimulus is presented
    at 3 positions equaly distributed along RF center at distance rf_jitter*rf_size.
    Returned array with responses is used in plots like neuron_activity_plot, complex_shapes_modulation.
    """
    topo.command.analysis.measure_sine_pref()
    topo.command.pylabplots.measure_cog()

    sheet = topo.sim[sheet_to_measure]

    (or_mat,or_box) = sheet.sheet_views['OrientationPreference'].view()
    (xcog_mat,xcog_box) = sheet.sheet_views['XCoG'].view()
    (ycog_mat,ycog_box) = sheet.sheet_views['YCoG'].view()

    sheet_density = int(sheet.density)
    sheet_neurons = sheet_density**2

    used_neurons = range(sheet_neurons)
    random.shuffle(used_neurons)
    used_neurons = used_neurons[:neurons_to_measure]

    i=0
    responses = []
    for x in range(sheet_density):
        for y in range(sheet_density):
            if((sheet_density*x + y) in used_neurons):
                print "Measuring neuron [%d,%d] - %d of %d" % (x,y,i+1,neurons_to_measure)
                i += 1
                subclass_responses = []
                for subclass in stimuli_subclasses:
                    pattern_responses = []
                    for pattern in subclass:
                        jitter = []
                        or_correction = or_mat[x,y]*numpy.pi
                        pattern.orientation += or_correction
                        pattern.x = xcog_mat[x,y] + rf_size*rf_jitter
                        pattern.y = ycog_mat[x,y]
                        topo.command.pattern_present({'Retina':pattern})
                        jitter.append(sheet.activity[x,y])
                        pattern.x = xcog_mat[x,y] - rf_size*rf_jitter/2
                        pattern.y = ycog_mat[x,y] + numpy.sqrt(numpy.power(rf_size*rf_jitter,2.0)-numpy.power(rf_size*rf_jitter/2,2.0))
                        topo.command.pattern_present({'Retina':pattern})
                        jitter.append(sheet.activity[x,y])
                        pattern.x = xcog_mat[x,y] - rf_size*rf_jitter/2
                        pattern.y = ycog_mat[x,y] - numpy.sqrt(numpy.power(rf_size*rf_jitter,2.0)-numpy.power(rf_size*rf_jitter/2,2.0))
                        topo.command.pattern_present({'Retina':pattern})
                        jitter.append(sheet.activity[x,y])
                        pattern_responses.append(numpy.mean(jitter))
                        pattern.orientation -= or_correction
                    subclass_responses.append(pattern_responses)
                responses.append(subclass_responses)
    return responses

class best_subclasses_plot(PylabPlotCommand):
    """
    Calculates subclass preferences. Each neuron is assigned to the subclass
    which contains its most effective stimulus.
    """
    def __call__(self, responses=[], stimuli_subclasses_labels=[], sign_level=0.1, **params):
        """
        Takes parameter responses as returned from measure_responses.  Labels must
        follow order of subclasses in array with responses. Level of significance
        is minimal difference between two best stimuli and determinates part of
        neuron ploted with different colour.
        """
        p=ParamOverrides(self,params)

        subclasses_length = max(map(len,responses))
        responses_best = map(lambda x: map(numpy.max, x), responses)
        responses_best_class = []
        for i in range(len(responses_best)):
            responses_best_class.append([(responses_best[i][j],j) for j in range(len(responses[i]))])
            responses_best_class[-1].sort(reverse=True)

        classes = map(operator.itemgetter(1),map(operator.itemgetter(0),responses_best_class))
        classes_count = []
        for i in range(max(map(len,responses))):
            classes_count.append(classes.count(i))

        for i in range(subclasses_length):
            print "%-20s (%d prefering, %.2f %%)" % (stimuli_subclasses_labels[i], classes_count[i], classes_count[i] * 100.0 / len(classes))

        responses_diff = []
        for i in range(len(responses_best_class)):
          responses_diff.append([(responses_best_class[i][j][0]-responses_best_class[i][j+1][0],responses_best_class[i][j][1],i) for j in range(subclasses_length-1)])

        best_diffs = map(operator.itemgetter(1), sorted(filter(lambda x: x[0] > sign_level, map(operator.itemgetter(0), responses_diff)), reverse=True))

        sign_classes_count = []
        for i in range(subclasses_length):
          sign_classes_count.append(best_diffs.count(i))

        pylab.figure()

        # the x locations for the groups
        ind = 0.1 + numpy.arange(subclasses_length)
        # the width of the bars
        width = 0.8

        ax = pylab.subplot(111)
        rects = ax.bar(ind, classes_count, width, color='#333ccc', alpha=0.8)
        ax.bar(ind, sign_classes_count, width, color='#ffff66', alpha=0.8)
        ax.set_ylabel('# neurons')
        ax.set_xticks(ind+width)
        ax.set_xticklabels(stimuli_subclasses_labels,rotation=45,ha='right')

        pylab.subplots_adjust(bottom=0.2)
        self._generate_figure(p)

class neuron_activity_plot(PylabPlotCommand):
    """
    Generates two figures. First contains color-coded responses for one neuron, second
    contains color-code legend and bar plot with average responses in subclasses.
    """
    def __call__(self, neuron_responses=[], stimuli_subclasses=[], stimuli_subclasses_labels=[], density=100, **params):
        """
        Takes responses of one measured neuron (as returned by measure_responses) and its set
        of measured stimuli. Labels must follow order of subclasses in array with responses.
        """
        p=ParamOverrides(self,params)

        stimuli_flat = sum(stimuli_subclasses,[])
        responses_flat = sum(neuron_responses,[])
        if(len(stimuli_flat) != len(responses_flat)):
            self.warning( "Responses and stimuli do not have same shape, that is weird." )
        stimuli_length = min(len(stimuli_flat),len(responses_flat))
        if(stimuli_length == 0):
            self.warning( "Responses or stimuli are empty, nothing to do." )
            return
        subclasses_length = len(neuron_responses)

        response_min = min(map(min, neuron_responses))
        response_max = max(map(max, neuron_responses))
        response_range = response_max - response_min

        pylab.figure()

        for pattern in range(stimuli_length):
            ax = pylab.subplot(4,int(numpy.ceil(float(stimuli_length)/4)),int(numpy.ceil(float(stimuli_length)/4))*(pattern%4)+(pattern/4)+1)
            pattern_response = responses_flat[pattern]
            clmax = pylab.cm.get_cmap()([(pattern_response-response_min)/response_range])[0]
            cdict = {'red':   [(0.0,  0.0, 0.0),
                               (1.0, clmax[0], clmax[0])],
                     'green': [(0.0,  0.0, 0.0),
                               (1.0, clmax[1], clmax[1])],
                     'blue':  [(0.0,  0.0, 0.0),
                               (1.0, clmax[2], clmax[2])]}
            bbox = BoundingBox(points=((-0.5,-0.5), (0.5,0.5)))
            ax.imshow(stimuli_flat[pattern](xdensity=density,ydensity=density,bounds=bbox),
                      cmap=matplotlib.colors.LinearSegmentedColormap('activity',cdict))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])

        self._generate_figure(p)

        pylab.figure()
        ax = pylab.subplot(2,1,1)
        ax.imshow([range(100)]*5)
        ax.set_yticklabels([])
        ax.set_xticklabels(["%.2f" % response_min,"%.2f" % (response_min+response_range)])
        ax.set_yticks([])
        ax.set_xticks([0,99])

        # the x locations for the groups
        ind = 0.1 + numpy.arange(subclasses_length)
        # the width of the bars
        width = 0.8

        subclass_means = [numpy.mean(neuron_responses[i]) for i in range(subclasses_length)]
        subclass_std = [numpy.std(neuron_responses[i]) for i in range(subclasses_length)]

        ax = pylab.subplot(2,1,2)
        rects = ax.bar(ind, subclass_means, width, color='#cccfff', alpha=0.5, yerr=subclass_std)
        ax.set_ylabel('average class activity')
        ax.set_ybound(0.0,1.0)
        ax.set_yticks(numpy.arange(0.0,2.0,0.2))
        ax.set_yticklabels(numpy.arange(0.0,2.0,0.2))
        ax.set_xticks(ind+width)
        ax.set_xticklabels(stimuli_subclasses_labels,ha='right',rotation=45)

        pylab.subplots_adjust(bottom=0.3)
        self._generate_figure(p)

    
class modulation_plot(PylabPlotCommand):
    """
    Complex shapes modulation plot. Axis x describes comples shapes selectivity index
    calculated as variance of best responses from complex subclasses. Axis y shows
    within preferred complex subclass index calculated as variance of responses in complex
    subclass with best response.
    """
    def __call__(self, responses=[], responses_classes_labels=[],
                 x_label="Complex Shape Selectivity (CSS) index",
                 y_label="Selectivity Within Preferred Complex Subclass", **params):
        """
        Parameter responses as returned by function measure_responses.
        Labels must follow order of subclasses in array with responses.
        """
        p=ParamOverrides(self,params)

        N = len(responses)
        responses_best = map(lambda x: map(numpy.max, x), responses)

        responses_best_class = []
        for i in range(N):
            responses_best_class.append([(responses_best[i][j],j,i) for j in range(len(responses_best[i]))])
            responses_best_class[-1].sort(reverse=True)

        css = []
        for i in range(N):
          css.append(numpy.var(map(operator.itemgetter(0),filter(lambda x: x[1] != 0,responses_best_class[i]))))

        wps = []
        for i in range(N):
          most_effective_non_cartesian_class = filter(lambda x: x[1] != 0,responses_best_class[i])[0][1]
          wps.append(numpy.var(responses[i][most_effective_non_cartesian_class]))

        left, width = 0.17, 0.6
        bottom, height = 0.17, 0.6

        bottom_h = bottom+height+0.02
        left_h = left+width+0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.15]
        rect_histy = [left_h, bottom, 0.15, height]

        pylab.figure()

        axScatter = pylab.axes(rect_scatter)
        axScatter.grid(True)
        axHistx = pylab.axes(rect_histx)
        axHisty = pylab.axes(rect_histy)

        axScatter.set_xlabel(x_label)
        axScatter.set_ylabel(y_label)

        axHisty.set_xlabel('# neurons')
        axHistx.set_ylabel('# neurons')

        # no labels
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        # the scatter plot:
        colours = ['b','g','r','c','m','#ff9966','k','0.80','0.50','y']
        if(responses_classes_labels == []):
            axScatter.scatter(css, wps)
        else:
            handles = []
            labels = []
            for subclass in range(len(responses_classes_labels)):
                neurons = map(operator.itemgetter(2), map(operator.itemgetter(0), filter(lambda x: x[0][1] == subclass,responses_best_class)))
                if(neurons != []):
                    x = [css[i] for i in neurons]
                    y = [wps[i] for i in neurons]
                    handles.append(axScatter.scatter(x, y, c=colours[subclass % len(colours)]))
                    labels.append(responses_classes_labels[subclass])
            axScatter.legend(handles, labels)

        # now determine nice limits by hand:
        xymax = numpy.max([numpy.max(css), numpy.max(wps)])
        binwidth = xymax / 50
        lim = (int(xymax/binwidth) + 1) * binwidth

        axScatter.set_xlim((0, lim))
        axScatter.set_ylim((0, lim))

        bins = numpy.arange(0, lim + binwidth, binwidth)
        axHistx.hist(css, bins=bins)
        axHisty.hist(wps, bins=bins, orientation='horizontal')

        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())

        self._generate_figure(p)

