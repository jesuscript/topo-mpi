import __main__
import numpy
import pylab
import os.path
import os
import copy
import pdb

import param
import topo
import topo.pattern.basic
import topo.command.analysis
from math import pi, sqrt, exp, pow
from numpy.oldnumeric import zeros, Float, sum
from topo.projection.basic import CFProjection, SharedWeightCFProjection
from topo.base.boundingregion import BoundingBox 
from topo.pattern.basic import Gaussian, Selector, Null
from topo.transferfn.basic import DivisiveNormalizeL1, HomeostaticMaxEnt, TransferFnWithState, Sigmoid, PiecewiseLinear
from topo.base.arrayutil import clip_lower
from topo.sheet.lissom import LISSOM
from topo.sheet.optimized import NeighborhoodMask_Opt, LISSOM_Opt
from topo.plotting.plotfilesaver import * 
from topo.command.pylabplots import cyclic_tuning_curve, matrixplot
from topo.command.analysis import save_plotgroup
from param import normalize_path
from topo.command.pylabplots import plot_tracked_attributes
from topo.base.functionfamily import CoordinateMapperFn
from topo.plotting.bitmap import MontageBitmap
from topo.base.patterngenerator import PatternGenerator, Constant 
from topo.transferfn.basic import  Sigmoid

class surround_analysis():

    peak_near_facilitation_hist = []
    peak_supression_hist  = []   
    peak_far_facilitation_hist  = []
    sheet_name = ""
    data_dict = {}
    
    low_contrast=100
    high_contrast=200
    
    def __init__(self,sheet_name="V1Complex"):
        from topo.analysis.featureresponses import MeasureResponseCommand, FeatureMaps, FeatureCurveCommand, UnitCurveCommand, SinusoidalMeasureResponseCommand,PatternPresenter
        import pylab
        self.sheet_name=sheet_name
        import topo
        self.sheet=topo.sim[sheet_name]
        # Center mask to matrixidx center
        self.center_r,self.center_c = self.sheet.sheet2matrixidx(0,0)
        self.center_x,self.center_y = self.sheet.matrixidx2sheet(self.center_r,self.center_c)
	
	
        from topo.analysis.featureresponses import PatternPresenter            
        PatternPresenter.duration=2.0
        import topo.command.pylabplots
        reload(topo.command.pylabplots)

	
        FeatureCurveCommand.curve_parameters=[{"contrast":self.low_contrast},{"contrast":self.high_contrast}]
        FeatureCurveCommand.display=True
        FeatureCurveCommand.sheet=topo.sim[sheet_name]
        SinusoidalMeasureResponseCommand.num_phase=8
        SinusoidalMeasureResponseCommand.frequencies=[__main__.__dict__.get('FREQ',2.4)]
        SinusoidalMeasureResponseCommand.scale=1.0
        MeasureResponseCommand.scale=1.0
        FeatureCurveCommand.num_orientation=12
        
       


    def analyse(self,steps=[],ns=10,offset_x=0,offset_y=0):
        
        save_plotgroup("Orientation Preference and Complexity")

        #save_plotgroup("Position Preference")
        for (x,y) in steps:
                xindex = self.center_r+offset_x+x
                yindex = self.center_c+offset_y+y
                xcoor,ycoor = self.sheet.matrixidx2sheet(xindex,yindex)
                topo.command.pylabplots.measure_size_response.instance(sheet=self.sheet,num_phase=4,num_sizes=ns,max_size=3.0,coords=[(xcoor,ycoor)])(coords=[(xcoor,ycoor)],frequencies=[__main__.__dict__.get('FREQ',2.4)])        
                
                self.data_dict[(xindex,yindex)] = {}
                self.data_dict[(xindex,yindex)]["ST"] = self.calculate_RF_sizes(xindex, yindex)
                self.plot_size_tunning(xindex,yindex)
                
                self.data_dict[(xindex,yindex)]["OCT"] = self.perform_orientation_contrast_analysis(self.data_dict[(xindex,yindex)]["ST"],xcoor,ycoor,xindex,yindex)
                self.plot_orientation_contrast_tuning(xindex,yindex)
                self.plot_orientation_contrast_tuning_abs(xindex,yindex)
                
                f = open(normalize_path("dict.dat"),'wb')
	        import pickle
        	pickle.dump(self.data_dict,f)
		f.close()
        
        self.plot_histograms_of_measures()
        lhi = compute_local_homogeneity_index(self.sheet.sheet_views['OrientationPreference'].view()[0]*pi,0.5)                
        self.plot_map_feature_to_surround_modulation_feature_correlations(lhi,"Local Homogeneity Index")
        self.plot_map_feature_to_surround_modulation_feature_correlations(self.sheet.sheet_views['OrientationSelectivity'].view()[0],"OrientationSelectivity")
        self.plot_map_feature_to_surround_modulation_feature_correlations(self.sheet.sheet_views['OrientationPreference'].view()[0]*numpy.pi,"OrientationPreference")

        

    def perform_orientation_contrast_analysis(self,data,xcoor,ycoor,xindex,yindex):
        curve_data={}
        hc_curve = data["Contrast = " + str(self.high_contrast) + "%" ]
        lc_curve = data["Contrast = " + str(self.low_contrast) + "%" ]
        
        topo.command.pylabplots.measure_or_tuning(num_phase=4,num_orientation=12,size=lc_curve["measures"]["peak_near_facilitation"],curve_parameters=[{"contrast":self.low_contrast}],display=True,coords=[(xcoor,ycoor)],frequencies=[__main__.__dict__.get('FREQ',2.4)])
        topo.command.pylabplots.cyclic_tuning_curve.instance(x_axis="orientation",coords=[(xcoor,ycoor)])
        
        curve_name_ort = "Contrast = " + str(self.low_contrast) + "%";
        
        ar = []
        ors = []
        for o in self.sheet.curve_dict['orientation'][curve_name_ort].keys():
            ar.append(self.sheet.curve_dict['orientation'][curve_name_ort][o].view()[0][xindex][yindex])
            ors.append(o)
            
        peak_or_response = max(ar)
        orr=ors[numpy.argmax(ar)]
        orr_ort = orr + (numpy.pi/2.0)
        
        curve_data["ORTC"]={}
        curve_data["ORTC"]["info"]={}
        curve_data["ORTC"]["info"]["pref_or"]=orr
        print "ORIENTATION:", orr 
        topo.command.pylabplots.measure_orientation_contrast(sizecenter=lc_curve["measures"]["peak_near_facilitation"],
                                                             orientation_center=orr,
                                                             sizesurround=4.0,
                                                             size=0.0,
                                                             display=True,
                                                             contrastcenter=self.low_contrast,
                                                             thickness=4.0-lc_curve["measures"]["peak_near_facilitation"],
                                                             num_phase=8,
							     frequencies=[__main__.__dict__.get('FREQ',2.4)],
                                                             curve_parameters=[{"contrastsurround":self.low_contrast},{"contrastsurround":self.high_contrast}],coords=[(xcoor,ycoor)])
        
        for curve_label in sorted(self.sheet.curve_dict['orientationsurround'].keys()):
            print curve_label
            curve_data[curve_label]={}
            curve_data[curve_label]["data"]=self.sheet.curve_dict['orientationsurround'][curve_label]
            curve_data[curve_label]["measures"]={}
            print self.sheet.curve_dict['orientationsurround'][curve_label].keys() , "\nAAA" , orr_ort," ", orr
            pref_or_resp=self.sheet.curve_dict['orientationsurround'][curve_label][orr].view()[0][xindex][yindex]
            
            
            cont_or_resp=self.sheet.curve_dict['orientationsurround'][curve_label][orr_ort].view()[0][xindex][yindex]
            
            
            if pref_or_resp != 0:
                curve_data[curve_label]["measures"]["or_suppression"]=(pref_or_resp-cont_or_resp)/pref_or_resp
            else: 
                curve_data[curve_label]["measures"]["or_suppression"]=-1.0
        
        
        hc_curve_name_orc = "Contrastsurround = " + str(self.high_contrast) + "%";
        lc_curve_name_orc = "Contrastsurround = " + str(self.low_contrast) + "%";
        
        hc_pref_or_resp=self.sheet.curve_dict['orientationsurround'][hc_curve_name_orc][orr].view()[0][xindex][yindex]
        hc_cont_or_resp=self.sheet.curve_dict['orientationsurround'][hc_curve_name_orc][orr_ort].view()[0][xindex][yindex]
        lc_pref_or_resp=self.sheet.curve_dict['orientationsurround'][lc_curve_name_orc][orr].view()[0][xindex][yindex]
        lc_cont_or_resp=self.sheet.curve_dict['orientationsurround'][lc_curve_name_orc][orr_ort].view()[0][xindex][yindex]

        
        curve_data["ORTC"]["data"]=self.sheet.curve_dict['orientation'][curve_name_ort]
        curve_data["ORTC"]["measures"]={}
        curve_data["ORTC"]["measures"]["colinear_hc_suppresion_index"] = (peak_or_response - hc_pref_or_resp) / peak_or_response 
        curve_data["ORTC"]["measures"]["colinear_lc_suppresion_index"] = (peak_or_response - lc_pref_or_resp) / peak_or_response
        curve_data["ORTC"]["measures"]["orcontrast_hc_suppresion_index"] = (peak_or_response - hc_cont_or_resp) / peak_or_response 
        curve_data["ORTC"]["measures"]["orcontrast_lc_suppresion_index"] = (peak_or_response - lc_cont_or_resp) / peak_or_response
        
        return curve_data 


    def calculate_RF_sizes(self,xindex, yindex):
        curve_data = {}
        hc_curve_name = "Contrast = " + str(self.high_contrast) + "%";
        lc_curve_name = "Contrast = " + str(self.low_contrast) + "%";
        for curve_label in [hc_curve_name,lc_curve_name]:
            curve = self.sheet.curve_dict['size'][curve_label]
            curve_data[curve_label] = {}
            curve_data[curve_label]["data"] = curve  
            
            x_values = sorted(curve.keys())
            y_values = [curve[key].view()[0][xindex, yindex] for key in x_values]

            #compute critical indexes in the size tuning curves
            curve_data[curve_label]["measures"]={}
            curve_data[curve_label]["measures"]["peak_near_facilitation_index"] = numpy.argmax(y_values)
            curve_data[curve_label]["measures"]["peak_near_facilitation"] = x_values[curve_data[curve_label]["measures"]["peak_near_facilitation_index"]]

            if(curve_data[curve_label]["measures"]["peak_near_facilitation"] < (len(y_values) - 1)):
                curve_data[curve_label]["measures"]["peak_supression_index"] = curve_data[curve_label]["measures"]["peak_near_facilitation_index"] + numpy.argmin(y_values[curve_data[curve_label]["measures"]["peak_near_facilitation_index"] + 1:]) + 1
                curve_data[curve_label]["measures"]["peak_supression"] = x_values[curve_data[curve_label]["measures"]["peak_supression_index"]]
                curve_data[curve_label]["measures"]["suppresion_index"] = (y_values[curve_data[curve_label]["measures"]["peak_near_facilitation_index"]] - y_values[curve_data[curve_label]["measures"]["peak_supression_index"]])/ y_values[curve_data[curve_label]["measures"]["peak_near_facilitation_index"]]  
                
            if(curve_data[curve_label]["measures"].has_key("peak_supression_index") and (curve_data[curve_label]["measures"]["peak_supression_index"] < (len(y_values) - 1))):
                curve_data[curve_label]["measures"]["peak_far_facilitation_index"] = curve_data[curve_label]["measures"]["peak_supression_index"] + numpy.argmax(y_values[curve_data[curve_label]["measures"]["peak_supression_index"] + 1:]) + 1
                curve_data[curve_label]["measures"]["peak_far_facilitation"] = x_values[curve_data[curve_label]["measures"]["peak_far_facilitation_index"]]
                curve_data[curve_label]["measures"]["counter_suppresion_index"] = (y_values[curve_data[curve_label]["measures"]["peak_far_facilitation_index"]] - y_values[curve_data[curve_label]["measures"]["peak_supression_index"]])/ y_values[curve_data[curve_label]["measures"]["peak_near_facilitation_index"]]
                
                
        curve_data[hc_curve_name]["measures"]["contrast_dependent_shift"]=curve_data[lc_curve_name]["measures"]["peak_near_facilitation"]/curve_data[hc_curve_name]["measures"]["peak_near_facilitation"]                 
        curve_data[lc_curve_name]["measures"]["contrast_dependent_shift"]=curve_data[lc_curve_name]["measures"]["peak_near_facilitation"]/curve_data[hc_curve_name]["measures"]["peak_near_facilitation"]
        return curve_data   
       

    def plot_size_tunning(self, xindex, yindex):
        fig = pylab.figure()
        #f = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, 3.0), ylim=(-0.1, 4.0))
        f = fig.add_subplot(111)
        pylab.title(self.sheet_name, fontsize=12)
        colors=['red','blue','green','purple','orange','black','yellow']
        
        measurment = self.data_dict[(xindex,yindex)]["ST"]
        i = 0
        for curve_label in measurment.keys():
            curve =  measurment[curve_label]["data"]
            x_values = sorted(curve.keys())
            y_values = [curve[key].view()[0][xindex, yindex] for key in x_values]
            
            f.plot(x_values, y_values, lw=3, color=colors[i])
            
                
            f.annotate('', xy=(measurment[curve_label]["measures"]["peak_near_facilitation"], y_values[measurment[curve_label]["measures"]["peak_near_facilitation_index"]]), xycoords='data',
            xytext=(-1, 20), textcoords='offset points', arrowprops=dict(facecolor='green', shrink=0.05))
    
    
            if measurment[curve_label]["measures"].has_key("peak_supression"):
                f.annotate('', xy=(measurment[curve_label]["measures"]["peak_supression"], y_values[measurment[curve_label]["measures"]["peak_supression_index"]]), xycoords='data',
                           xytext=(-1, 20), textcoords='offset points', arrowprops=dict(facecolor='red', shrink=0.05))
            
            if measurment[curve_label]["measures"].has_key("peak_far_facilitation"):
                f.annotate('', xy=(measurment[curve_label]["measures"]["peak_far_facilitation"], y_values[measurment[curve_label]["measures"]["peak_far_facilitation_index"]]), xycoords='data',
                           xytext=(-1, 20), textcoords='offset points', arrowprops=dict(facecolor='blue', shrink=0.05))
            i+=1
            
        release_fig("STC[" + str(xindex) + "," + str(yindex) + "]")


    def plot_orientation_contrast_tuning_abs(self, xindex, yindex):
        fig = pylab.figure()
        f = fig.add_subplot(111, autoscale_on=True)
        pylab.title(self.sheet_name, fontsize=12)
        colors=['red','blue','green','purple','orange','black','yellow']
        orientation = self.data_dict[(xindex,yindex)]["OCT"]["ORTC"]["info"]["pref_or"]
        
        print orientation
        measurment = self.data_dict[(xindex,yindex)]["OCT"]
        i = 0
        for curve_label in measurment.keys():
            curve =  measurment[curve_label]["data"]
            x_values = sorted(curve.keys())
            y_values = [curve[key].view()[0][xindex, yindex] for key in x_values]
	    
	    print "ADSAAAAAAAAAAAAAAAAAAADASDA"
	    print x_values
	    print y_values

            f.plot(x_values, y_values, lw=3, color=colors[i])
            f.axvline(x=orientation,linewidth=4, color='r')
            i+=1
        
        release_fig("AbsOCTC[" + str(xindex) + "," + str(yindex) + "]")

    def plot_orientation_contrast_tuning(self, xindex, yindex):
        fig = pylab.figure()
        f = fig.add_subplot(111, autoscale_on=True)
        pylab.title(self.sheet_name, fontsize=12)
        colors=['red','blue','green','purple','orange','black','yellow']
        
        orientation = self.data_dict[(xindex,yindex)]["OCT"]["ORTC"]["info"]["pref_or"]
        
        measurment = self.data_dict[(xindex,yindex)]["OCT"]
        i = 0
        hc_curve_name_orc = "Contrastsurround = " + str(self.high_contrast) + "%";
        
        for curve_label in measurment.keys():
            
            print "AAA:",curve_label
            curve =  measurment[curve_label]["data"]
            x_values = sorted(curve.keys())
            y_values = []
            for k in x_values:
                y_values.append(curve[k].view()[0][xindex, yindex])
                
            
            
            x_values=numpy.array(x_values)-orientation
	    
	    print x_values

            for j in xrange(0,len(x_values)):
                if x_values[j] > numpy.pi/2.0:
                   x_values[j] -= numpy.pi 
                if x_values[j] < -numpy.pi/2.0:
                   x_values[j] += numpy.pi

            for j in xrange(0,len(x_values)):
                if x_values[j] > numpy.pi/2.0:
                   x_values[j] -= numpy.pi 
                if x_values[j] < -numpy.pi/2.0:
                   x_values[j] += numpy.pi


            inds = numpy.argsort(x_values)
	    y_values = numpy.take(y_values, inds)
            x_values = sorted(x_values)

            numpy.append(y_values,y_values[0])
            numpy.append(x_values,x_values[0]+numpy.pi)
            
            f.plot(x_values, y_values, lw=3, color=colors[i],label=curve_label)
            i+=1
        
	pylab.legend(loc='lower left')
        release_fig("OCTC[" + str(xindex) + "," + str(yindex) + "]")
        
        fig = pylab.figure()
        f = fig.add_subplot(111, autoscale_on=True)

        curve =  measurment["ORTC"]["data"]
        x_values= sorted(curve.keys())
        y_values=[curve[key].view()[0][xindex,yindex] for key in x_values]

        f.plot(x_values, y_values, lw=3)
        release_fig("OTC[" + str(xindex) + "," + str(yindex) + "]")

        

    def plot_map_feature_to_surround_modulation_feature_correlations(self,map_feature,map_feature_name):
        
        from numpy import polyfit
        
        raster_plots_lc={}
        raster_plots_hc={}
        for (xcoord,ycoord) in self.data_dict.keys():
            for curve_type in self.data_dict[(xcoord,ycoord)].keys():
                print curve_type
                if curve_type == "ST":
                   curve_label = "Contrast"
                else:
                   curve_label = "Contrastsurround" 
                
                print self.data_dict[(xcoord,ycoord)][curve_type].keys()
                
                if self.data_dict[(xcoord,ycoord)][curve_type].has_key(curve_label + " = " + str(self.low_contrast) + "%"):
                    for measure_name in self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.high_contrast) + "%"]["measures"].keys():
                        if not raster_plots_hc.has_key(measure_name):
                            raster_plots_hc[measure_name]=[[],[]]    
                        raster_plots_hc[measure_name][0].append(self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.high_contrast) + "%"]["measures"][measure_name])
                        raster_plots_hc[measure_name][1].append(map_feature[xcoord,ycoord])        
                
                if self.data_dict[(xcoord,ycoord)][curve_type].has_key(curve_label + " = " + str(self.low_contrast) + "%"):
                    for measure_name in self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.low_contrast) + "%"]["measures"].keys():
                        if not raster_plots_lc.has_key(measure_name):
                            raster_plots_lc[measure_name]=[[],[]]    
                        raster_plots_lc[measure_name][0].append(self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = "  + str(self.low_contrast) + "%"]["measures"][measure_name])
                        raster_plots_lc[measure_name][1].append(map_feature[xcoord,ycoord])        

        for key in raster_plots_hc.keys():
                fig = pylab.figure()
                f = fig.add_subplot(111)
                f.set_xlabel(str(key))
                f.set_ylabel(map_feature_name)
                try:
                    #correlation = numpy.corrcoef(raster_plots_hc[key][0],raster_plots_hc[key][1])[0,1]
		    import scipy.stats
		    correlation = scipy.stats.pearsonr(raster_plots_hc[key][0],raster_plots_hc[key][1])[0]
		    pval= scipy.stats.pearsonr(raster_plots_hc[key][0],raster_plots_hc[key][1])[1] 
                except FloatingPointError:
                      correlation = 0
                m,b = numpy.polyfit(raster_plots_hc[key][0],raster_plots_hc[key][1],1)
                f.plot(raster_plots_hc[key][0],raster_plots_hc[key][1],'ro')
                f.plot(raster_plots_hc[key][0],m*numpy.array(raster_plots_hc[key][0])+b,'-k',linewidth=2)
                release_fig("RasterHC<" + map_feature_name + ","+ key +  " Corr:"+ str(correlation) + '|'+ str(pval) + ">")
                

        for key in raster_plots_lc.keys():
                fig = pylab.figure()
                f = fig.add_subplot(111)
                f.set_xlabel(str(key))
                f.set_ylabel(map_feature_name)
                m,b = numpy.polyfit(raster_plots_lc[key][0],raster_plots_lc[key][1],1)
                try:
                    #correlation = numpy.corrcoef(raster_plots_lc[key][0],raster_plots_lc[key][1])[0,1]
		    import scipy.stats
		    correlation = scipy.stats.pearsonr(raster_plots_hc[key][0],raster_plots_hc[key][1])[0]
		    pval= scipy.stats.pearsonr(raster_plots_hc[key][0],raster_plots_hc[key][1])[1] 
                except FloatingPointError:
                      correlation = 0
                f.plot(raster_plots_lc[key][0],raster_plots_lc[key][1],'ro')
                f.plot(raster_plots_lc[key][0],m*numpy.array(raster_plots_lc[key][0])+b,'-k',linewidth=2)
                release_fig("RasterLC<" + map_feature_name + ","+ key + " Corr:"+ str(correlation)+ '|'+ str(pval) + ">")

            
    def plot_histograms_of_measures(self):
        histograms_lc = {} 
        histograms_hc = {}
        for (xcoord,ycoord) in self.data_dict.keys():
            for curve_type in self.data_dict[(xcoord,ycoord)].keys():
                print curve_type
                if curve_type == "ST":
                   curve_label = "Contrast"
                else:
                   curve_label = "Contrastsurround"
                print self.data_dict[(xcoord,ycoord)][curve_type].keys()   
                for measure_name in self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.high_contrast) + "%"]["measures"].keys():
                    if not histograms_hc.has_key(curve_type + "_" + measure_name):
                        histograms_hc[curve_type + "_" + measure_name]=[]
                    histograms_hc[curve_type + "_" + measure_name].append(self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.high_contrast) + "%"]["measures"][measure_name])

                for measure_name in self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.low_contrast) + "%"]["measures"].keys():
                    if not histograms_lc.has_key(curve_type + "_" + measure_name):
                        histograms_lc[curve_type + "_" + measure_name]=[]
                    histograms_lc[curve_type + "_" + measure_name].append(self.data_dict[(xcoord,ycoord)][curve_type][curve_label + " = " + str(self.low_contrast) + "%"]["measures"][measure_name])
                
        for key in histograms_lc.keys():
                if ((len(histograms_lc[key]) != 0) and (len(histograms_hc[key]) != 0)):
                    fig = pylab.figure()
                    pylab.title(self.sheet_name+ " " + "MeanLC: " + str(numpy.mean(histograms_lc[key])) + "+/-" + str(numpy.std(histograms_lc[key])/ (len(histograms_lc[key])*len(histograms_lc[key]))) + "MeanHC: " + str(numpy.mean(histograms_hc[key])) + "+/-" + str(numpy.std(histograms_hc[key])/ (len(histograms_hc[key])*len(histograms_hc[key]))) , fontsize=12)
                    
                    f = fig.add_subplot(111)
                    f.set_xlabel(str(key))
                    f.set_ylabel('#Cells')
                    mmax = numpy.max(numpy.max(histograms_lc[key]),numpy.max(histograms_lc[key]))
                    mmin = numpy.min(numpy.min(histograms_lc[key]),numpy.min(histograms_lc[key]))
                    bins = numpy.arange(mmin-0.01,mmax+0.01,(mmax+0.01-(mmin-0.01))/10.0)
                    f.hist(histograms_lc[key],bins=bins,normed=False)
                    #f.axvline(x=numpy.mean(histograms_lc[key]),linewidth=4, color='r')
                    release_fig("Histogram<" + key + ">")
                    print len(histograms_lc[key])
                    print key + "LC mean :" + str(numpy.mean(histograms_lc[key]))
                    print key + "HC mean :" + str(numpy.mean(histograms_hc[key]))
                else:
                    print "Histogram ", key , " empty!"

def compute_local_homogeneity_index(or_map,sigma):
    (xsize,ysize) = or_map.shape 
    
    lhi = numpy.zeros(or_map.shape) 
    
    for sx in xrange(0,xsize):
        for sy in xrange(0,ysize):
            lhi_current=[0,0]
            for tx in xrange(0,xsize):
                for ty in xrange(0,ysize):
                    lhi_current[0]+=numpy.exp(-((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty))/(2*sigma*sigma))*numpy.cos(2*or_map[tx,ty])
                    lhi_current[1]+=numpy.exp(-((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty))/(2*sigma*sigma))*numpy.sin(2*or_map[tx,ty])
            # print sx,sy
            # print lhi.shape
            # print lhi_current        
            lhi[sx,sy]= numpy.sqrt(lhi_current[0]*lhi_current[0] + lhi_current[1]*lhi_current[1])/(2*numpy.pi*sigma*sigma)
                    
    return lhi       

def release_fig(filename=None):
    import pylab        
    pylab.show._needmain=False
    if filename is not None:
       fullname=filename+str(topo.sim.time())+".png"
       pylab.savefig(normalize_path(fullname))
    else:
       pylab.show()



def plot_neural_dynamics(sheet_names,neurons,pattern_generator,prefix):
    """
    call example : contrib.surround_analysis.plot_neural_dynamics(["V1"],[("V1",(0.0,0.0)),("V1",(0.1,0.1))])
    """
    
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

    for i in xrange(0,85):
        pp = PatternPresenter(pattern_generator=pattern_generator,duration=i*0.05,contrast_parameter="weber_contrast")
        
        for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
        topo.sim.state_push()
        for f in PatternDrivenAnalysis.pre_presentation_hooks: f()
        
        pp({},{})
        topo.guimain.refresh_activity_windows()
        
        for key in sheet_names:
            for i in topo.sim[key].projections().keys():
                data[key][i].append(topo.sim[key].projections()[i].activity.copy())
            data[key]["act"].append(topo.sim[key].activity.copy())

        for f in PatternDrivenAnalysis.post_presentation_hooks: f()
        topo.sim.state_pop()
        for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
        

    for n in neurons:
       (sheetname, (x,y)) = n
       (xx,yy) = topo.sim[sheetname].sheet2matrixidx(x,y)
       pylab.figure()
       pylab.title(prefix+sheetname+" [" + str(x) + "," +str(y) + "]")
       for projname in data[sheetname].keys():
           a = []
           for act in data[sheetname][projname]:
               a.append(act[xx,yy])
           pylab.plot(a,label=projname)
       pylab.legend(loc='upper left')
       release_fig(prefix+"_"+sheetname+"_settling")
    pylab.show._needmain=False
    #release_fig()
    pylab.show()
     
     
     
     
def run_dynamics_analysis(x,y,cs,scale):
    from topo.pattern.basic import OrientationContrast
    
    (xx,yy) = topo.sim["V1Complex"].sheet2matrixidx(x,y)
    
    orr=numpy.pi*topo.sim["V1Complex"].sheet_views["OrientationPreference"].view()[0][xx][yy]
    
    pg = OrientationContrast(orientationcenter=orr,orientationsurround=orr,sizecenter=cs,sizesurround=2.0,thickness=2.0-cs-0.2,scalecenter=scale,scalesurround=scale,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))
    plot_neural_dynamics(["V1Complex","V1ComplexInh"],[("V1Complex",(x,y)),("V1ComplexInh",(x,y))],pg,"colinear")
    
    pg = OrientationContrast(orientationcenter=orr,orientationsurround=orr+numpy.pi/2,sizecenter=cs,sizesurround=2.0,thickness=2.0-cs-0.2,scalecenter=scale,scalesurround=scale,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))
    
    plot_neural_dynamics(["V1Complex","V1ComplexInh"],[("V1Complex",(x,y)),("V1ComplexInh",(x,y))],pg,"ortogonal")

def test(x,y,cs,scale):
    from topo.pattern.basic import OrientationContrast
    from topo.command.basic import pattern_present
    from topo.base.functionfamily import PatternDrivenAnalysis
    from topo.pattern.basic import OrientationContrast
    from topo.analysis.featureresponses import PatternPresenter
    from topo.base.sheet import Sheet

    (xx,yy) = topo.sim["V1Complex"].sheet2matrixidx(x,y)
    
    orr=numpy.pi*topo.sim["V1Complex"].sheet_views["OrientationPreference"].view()[0][xx][yy]
    
    pg1 = OrientationContrast(orientationcenter=orr,orientationsurround=orr,sizecenter=cs,sizesurround=2.0,thickness=2.0-cs,scalecenter=scale,scalesurround=scale,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))

    pg2 = OrientationContrast(orientationcenter=orr,orientationsurround=orr+numpy.pi/2,sizecenter=cs,sizesurround=2.0,thickness=2.0-cs,scalecenter=scale,scalesurround=scale,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))

    pp = PatternPresenter(pattern_generator=pg1,duration=4.0,contrast_parameter="weber_contrast")
        
    for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
    topo.sim.state_push()
    for f in PatternDrivenAnalysis.pre_presentation_hooks: f()
        
    pp({},{})
    topo.guimain.refresh_activity_windows()
    
    LongEI = topo.sim["V1ComplexInh"].projections()["LongEI"].cfs[xx,yy].weights
    LongEIAct = topo.sim["V1ComplexInh"].projections()["LongEI"].activity[xx,yy]
    V1CActivity = topo.sim["V1Complex"].activity

    for f in PatternDrivenAnalysis.post_presentation_hooks: f()
    topo.sim.state_pop()
    for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
	
    pp = PatternPresenter(pattern_generator=pg2,duration=4.0,contrast_parameter="weber_contrast")
        
    for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
    topo.sim.state_push()
    for f in PatternDrivenAnalysis.pre_presentation_hooks: f()
        
    pp({},{})
    topo.guimain.refresh_activity_windows()
    
    LongEIOrt = topo.sim["V1ComplexInh"].projections()["LongEI"].cfs[xx,yy].weights
    LongEIActOrt = topo.sim["V1ComplexInh"].projections()["LongEI"].activity[xx,yy]
    V1CActivityOrt = topo.sim["V1Complex"].activity

    for f in PatternDrivenAnalysis.post_presentation_hooks: f()
    topo.sim.state_pop()
    for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
	
    print numpy.shape(LongEIOrt)
    print numpy.shape(V1CActivityOrt)
    
    print "colinear:", LongEIAct
    print "orthogonal:", LongEIActOrt
    
    print "cross colinear:", numpy.sum(numpy.multiply(numpy.mat(LongEI),numpy.mat(V1CActivity)))
    print "cross orthogonal:", numpy.sum(numpy.multiply(numpy.mat(LongEIOrt),numpy.mat(V1CActivityOrt)))
    
    
    pylab.figure()
    pylab.subplot(1,3,1)
    pylab.imshow(LongEI)
    pylab.subplot(1,3,2)
    pylab.imshow(V1CActivity)
    pylab.subplot(1,3,3)
    pylab.imshow(numpy.array(numpy.multiply(numpy.mat(LongEI),numpy.mat(V1CActivity))))
		
    pylab.figure()
    pylab.subplot(1,3,1)
    pylab.imshow(LongEIOrt)
    pylab.subplot(1,3,2)
    pylab.imshow(V1CActivityOrt)
    pylab.subplot(1,3,3)
    pylab.imshow(numpy.array(numpy.multiply(numpy.mat(LongEIOrt),numpy.mat(V1CActivityOrt))))
	
		
from topo.pattern.basic import SineGrating, Disk
class SineGratingDiskTemp(SineGrating):
      mask_shape = param.Parameter(default=Disk(smoothing=0,size=1.0))
		
		
def size_tining_activity_evolution(x,y,cs,scale):
    from topo.pattern.basic import OrientationContrast
    from topo.command.basic import pattern_present
    from topo.base.functionfamily import PatternDrivenAnalysis
    from topo.pattern.basic import OrientationContrast
    from topo.analysis.featureresponses import PatternPresenter
    from topo.base.sheet import Sheet

    (xx,yy) = topo.sim["V1Complex"].sheet2matrixidx(x,y)
    orr=numpy.pi*topo.sim["V1Complex"].sheet_views["OrientationPreference"].view()[0][xx][yy]
    
    pg1 = SineGratingDiskTemp(orientation=orr,phase=4.5,size=0.3,scale=3.0,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))
    pg2 = SineGratingDiskTemp(orientation=orr,phase=4.5,size=0.8,scale=scale,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))
    pg3 = SineGratingDiskTemp(orientation=orr,phase=4.5,size=1.1,scale=scale,x=x,y=y,frequency=__main__.__dict__.get('FREQ',2.4))
    
    pp = PatternPresenter(pattern_generator=pg1,duration=4.0,contrast_parameter="weber_contrast")
        
    for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
    topo.sim.state_push()
    for f in PatternDrivenAnalysis.pre_presentation_hooks: f()

    pp({},{})
    topo.guimain.refresh_activity_windows()
    
    V1CActivity1 = topo.sim["V1Complex"].activity.copy()

    for f in PatternDrivenAnalysis.post_presentation_hooks: f()
    topo.sim.state_pop()
    for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
	
    pp = PatternPresenter(pattern_generator=pg2,duration=4.0,contrast_parameter="weber_contrast")
        
    for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
    topo.sim.state_push()
    for f in PatternDrivenAnalysis.pre_presentation_hooks: f()
        
    pp({},{})
    topo.guimain.refresh_activity_windows()
    
    V1CActivity2 = topo.sim["V1Complex"].activity.copy()

    for f in PatternDrivenAnalysis.post_presentation_hooks: f()
    topo.sim.state_pop()
    for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
	
    pp = PatternPresenter(pattern_generator=pg3,duration=4.0,contrast_parameter="weber_contrast")
        
    for f in PatternDrivenAnalysis.pre_analysis_session_hooks: f()
    topo.sim.state_push()
    for f in PatternDrivenAnalysis.pre_presentation_hooks: f()
        
    pp({},{})
    topo.guimain.refresh_activity_windows()
    
    V1CActivity3 = topo.sim["V1Complex"].activity.copy()

    for f in PatternDrivenAnalysis.post_presentation_hooks: f()
    topo.sim.state_pop()
    for f in PatternDrivenAnalysis.post_analysis_session_hooks: f()
    
    pylab.figure()
    pylab.subplot(1,3,1)
    pylab.imshow(V1CActivity1,vmin=0.1,vmax=1.3,interpolation='nearest')
    pylab.ylabel('L2/3 activity',fontsize=30)
    pylab.xlabel('size = 0.15',fontsize=25)
    pylab.xticks([], [])  
    pylab.yticks([], [])
    pylab.colorbar(shrink=0.4)
    pylab.subplot(1,3,2)
    pylab.imshow(V1CActivity2,vmin=0.1,vmax=1.3,interpolation='nearest')
    pylab.xticks([], [])
    pylab.yticks([], [])
    pylab.xlabel('size = 0.8',fontsize=25)
    pylab.colorbar(shrink=0.4)
    pylab.subplot(1,3,3)
    pylab.imshow(V1CActivity3,vmin=0.1,vmax=1.3,interpolation='nearest')
    pylab.xticks([], [])
    pylab.yticks([], [])
    pylab.xlabel('size = 1.1',fontsize=25)
    pylab.colorbar(shrink=0.4)
    
    
	
				
