from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from pylab import *
from matplotlib.widgets import Button
from matplotlib import pyplot as P
from matplotlib.font_manager import FontProperties
import sys, getopt
import numpy as np
from xml.dom import minidom as md, ext 


def usage():
    print "Usage: python draw_component_scaling_graph.py [options] file..."
    print "Options:"
    print "  --help \t\tDisplay this information"
    print "  -s <param> \tUse <param> to plot a speed-up graph"
    print "  -c <params> \tUse comma-separated <params> to plot a component scaling graph"
    print "  -t <param> \tUse <param> to draw the Total line on a component scaling graph"
    print "  -l <label> \tAdd <label> to the caption of the graph"
    print "  -x <param> \tUse <param> as the X axis"
    print "  -z <param> \tUse <param> as the Z axis"




try:
    opts, args = getopt.getopt(sys.argv[1:],'c:t:s:l:x:z:',['help'])
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    usage()
    sys.exit(2)


if len(args) < 1:
    print "No filename supplied!\n"
    usage()
    sys.exit(2)
elif len(args) > 1:
    print "Only one file can be used for plotting!\n"
    usage()
    sys.exit(2)

x_label=None
z_label=None
c_labels=None
t_label=None
s_label=None
graph_label=None

for o, a in opts:
    if o == "--help":
        usage()
        sys.exit()
    elif o == "-x":
        x_label = a
    elif o == "-z":
        z_label = a
    elif o == "-c":
        #TODO: add space cleanup
        c_labels = a.replace(' ','').split(',')
    elif o == "-t":
        t_label = a
    elif o == "-s":
        s_label = a
    elif o == "-l":
        graph_label = a


if (c_labels!=None and s_label!=None):
    print "-c and -s options are mutually exclusive. Please use only one\n"
    usage()
    sys.exit(2)
    
if (t_label!=None and s_label!=None):
    print "-t and -s options are mutually exclusive. Please use only one\n"
    usage()
    sys.exit(2)
        
if (t_label==None and s_label==None and c_labels==None):
    print "At least one of -s, -t or -c options have to be used\n"
    usage()
    sys.exit(2)



xmldoc = md.parse(args[0])

# {"k":[v,...]},{"k2": [v2,...]}




if z_label!=None:
    distinct_z_vals = []
    for node in xmldoc.getElementsByTagName("run"):
        if node.getAttribute(z_label) not in distinct_z_vals:
            distinct_z_vals.append(node.getAttribute(z_label))


colors_used = []
cellText = []

fontP = FontProperties()
fontP.set_size('small')
colors = ["#FF0000","#00FF00","#0000FF","#AAAAAA","#33FFAA","#FF00FF","#AA33FF","#FFFF00"]    


i = 0
while 1:
    data = xmldoc.getElementsByTagName("run")
    if z_label!=None:
        data = [item for item in data if item.getAttribute(z_label)==distinct_z_vals[i]]

    if x_label!=None:
        distinct_x_vals = []
        tmp_data = []
        for node in data: # clearing up duplicates along the x axis
            if int(node.getAttribute(x_label)) not in distinct_x_vals:
                distinct_x_vals.append(int(node.getAttribute(x_label)))
                tmp_data.append(node)
        data = tmp_data
        data = sorted(data,key=lambda node: float(node.getAttribute(x_label)))
        distinct_x_vals = sort(distinct_x_vals)


    if x_label!=None:
        X = distinct_x_vals 
    else:
        X = range(1,len(data)+1)


    #(re-)init the plotter
    fig = figure(i)
    host = SubplotHost(fig,111)
    fig.add_subplot(host)

    #speedup graph
    Y = []
    if s_label!=None:
        for d in data:
            Y.append(float(d.getElementsByTagName(s_label)[0].childNodes[0].nodeValue))
        Y = [Y[0]/y for y in Y]
        host.plot(X,Y,color="black",label=str(data[0].getElementsByTagName(s_label)[0].getAttribute("name")),lw=3)
        host.set_ylabel("Speedup")
    else:
        #components graph
        comps=[]
        if c_labels!=None:
            for d in data:
                comps.append([])
                inc = 0.0
                for c in c_labels:
                    comps[-1].append(float(d.getElementsByTagName(c)[0].childNodes[0].nodeValue) + inc)
                    inc+=float(d.getElementsByTagName(c)[0].childNodes[0].nodeValue)
                comps[-1].append(0.0)
                    
            comps = map(lambda *row:list(row),*comps) # transposing comps

            for j in range(len(comps)-1):
                host.fill_between(X,comps[j],comps[j-1],facecolor=colors[j%len(colors)])
                host.plot(X,comps[j],'o-',color=colors[j%len(colors)],
                          label=data[0].getElementsByTagName(c_labels[j])[0].getAttribute("name"))
                
        #total line graph
        if t_label!=None:
            for d in data:
                Y.append(float(d.getElementsByTagName(t_label)[0].childNodes[0].nodeValue))
            host.plot(X,Y,color="black",label=str(data[0].getElementsByTagName(t_label)[0].getAttribute("name")),lw=3)

        host.set_ylabel("Time")

    host.set_xlabel(x_label)


    P.xlim([X[0],X[-1]])
    P.xticks(X)
    
    #reversing the order of labels
    handles, labels = host.get_legend_handles_labels()

    host.legend(handles[::-1],labels[::-1],loc="upper center",prop=fontP)

    host.toggle_axisline(False)
    host.grid(True)

    if graph_label==None:
        graph_label=args[0]


    if z_label==None:
        ttext = title(graph_label, fontsize=20)
        P.savefig(graph_label+".png")
    else:
        ttext = title(graph_label+", " + z_label+" = " + distinct_z_vals[i], fontsize=20)
        P.savefig(graph_label+"."+distinct_z_vals[i]+".png")

    i+=1
    if i>=len(distinct_z_vals):
        break
