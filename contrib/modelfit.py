from topo.pattern.basic import Gabor, SineGrating, Gaussian
import __main__
import numpy
import pylab
import matplotlib
from numpy import array, size, mat, shape, ones, arange
from topo import numbergen
#from topo.base.functionfamily import IdentityTF
from topo.transferfn.misc import PatternCombine
from topo.transferfn.misc import AttributeTrackingTF
from topo.transferfn.misc import HalfRectify
from topo.transferfn.basic import PiecewiseLinear, DivisiveNormalizeL1, IdentityTF, ActivityAveragingTF, Sigmoid
from topo.base.cf import CFSheet
from topo.projection.basic import CFProjection, SharedWeightCFProjection
from fixedpoint import FixedPoint
import topo

from topo.sheet import GeneratorSheet 
from topo.base.boundingregion import BoundingBox
from topo.pattern.image import FileImage
import contrib.jacommands
import contrib.dd
from matplotlib.ticker import MaxNLocator
from contrib.JanA.noiseEstimation import signal_power_test
from helper import *
from contrib.JanA.dataimport import sortOutLoading

#dd = contrib.dd.DB()
#dd.load_db("modelfitDB.dat")

save_fig=__main__.__dict__.get('SaveFig',False)
save_fig_directory='./'

def release_fig(filename=None):
    import pylab        
    if save_fig:
       pylab.savefig(save_fig_directory+filename)


class ModelFit():
    weigths = []
    retina_diameter=1.2
    density=24
    epochs = 500
    learning_rate = 0.00001
    DC = 0
    reliable_indecies=[]
    momentum=0.0
    num_of_units=0
    
    def init(self):
        self.reliable_indecies = ones(self.num_of_units)

    def calculateModelOutput(self,inputs,index):
        return self.weigths*inputs[index].T+self.DC.T

    def trainModel(self,inputs,activities,validation_inputs,validation_activities,stop=None):
        
        if stop==None:
           stop = numpy.ones(numpy.shape(activities[0])).copy()*1000000000000000000000
        
        delta=[]
        self.DC=numpy.array(activities[0]).copy()*0.0

        #self.weigths=numpy.mat(numpy.zeros((size(activities,1),size(inputs[0],1))))
	self.weigths=numpy.mat(numpy.identity(size(inputs[0],1)))
        best_weights = self.weigths.copy()

        mean_error=numpy.mat(numpy.zeros(shape(activities[0].T)))
        first_val_error=0
        val_err=0
        min_err=1000000000
        min_val_err=1000000000000
        min_val_err_array = ones(self.num_of_units)*10000000000000000000
        first_val_err_array = []
        err_hist = []
        val_err_hist = []
	
        val_pve = []
	
	validation_error=numpy.mat(numpy.zeros(shape(activities[0].T)))
	variance=numpy.mat(numpy.zeros(shape(activities[0].T)))
	for i in xrange(0,len(validation_inputs)):
                error = ((validation_activities[i].T - self.weigths*validation_inputs[i].T - self.DC.T))
                validation_error=validation_error+numpy.power(error,2)
		variance = variance + numpy.power((validation_activities[i] - numpy.mean(validation_activities,axis=0)),2).T
		
	print "INITIAL VALIDATION ERROR", numpy.sum(validation_error)/len(validation_inputs)/len(validation_error)
	print "INITIAL FEV on validation set", numpy.mean(1.0-numpy.divide(validation_error,variance))   
        
        for k in xrange(0,self.epochs):
            
            stop_learning = (stop>k)*1.0
            sl = numpy.mat(stop_learning).T
            for i in xrange(1, size(inputs[0],1)):
                sl = numpy.concatenate((sl,numpy.mat(stop_learning).T),axis=1)
            
            mean_error=numpy.mat(numpy.zeros(shape(activities[0].T)))
            validation_error=numpy.mat(numpy.zeros((len(activities[0].T),1)))
            tmp_weigths=numpy.mat(numpy.zeros((size(activities,1),size(inputs[0],1))))
            for i in xrange(0,len(inputs)):
                error = ((activities[i].T - self.weigths*inputs[i].T - self.DC.T))
                
                tmp_weigths = tmp_weigths + (error * inputs[i])
                mean_error=mean_error+numpy.power(error,2)
            
            err_hist.append(mean_error)
            
            if k == 0:
               delta = tmp_weigths/numpy.sqrt(numpy.sum(numpy.power(tmp_weigths,2)))
            else:
               delta = self.momentum*delta + (1.0-self.momentum)*tmp_weigths/numpy.sqrt(numpy.sum(numpy.power(tmp_weigths,2)))
            
            delta = numpy.multiply(delta/numpy.sqrt(numpy.sum(numpy.power(delta,2))),sl)
                   
            self.weigths = self.weigths + self.learning_rate*delta
            err = numpy.sum(mean_error)/len(inputs)/len(mean_error)    
            
            for i in xrange(0,len(validation_inputs)):
                error = ((validation_activities[i].T - self.weigths*validation_inputs[i].T - self.DC.T))
                validation_error=validation_error+numpy.power(error,2)
            val_err_hist.append(validation_error)
            val_err = numpy.sum(validation_error)/len(validation_inputs)/len(validation_error)    
            
               
            if val_err < min_val_err:
               min_val_err = val_err
            if err < min_err:
               min_err = err
               
            for i in xrange(0,len(min_val_err_array)):
                if min_val_err_array[i] > validation_error[i,0]:
                   min_val_err_array[i] = validation_error[i,0]
                   #!!!!!!!!!!!!!
                   best_weights[i,:] = self.weigths[i,:]
                
            if k == 0:
               first_val_error=val_err
               first_val_err_array = min_val_err_array.copy()

            print (k,err,val_err)
        print "First val error:" + str(first_val_error) + "\n Minimum val error:" + str(min_val_err) +        "\n Last val error:" + str(val_err) + "\nImprovement:" + str((first_val_error - min_val_err)/first_val_error * 100) + "%" #+ "\nBest cell by cell error:" + str(numpy.sum(min_val_err_array)/len(min_val_err_array)/len(validation_inputs)) + "\nBest cell by cell error improvement:" + str((first_val_err_array - min_val_err_array)/len(validation_inputs)/first_val_err_array)

        # plot error evolution
        a = err_hist[0].T/self.epochs
        b = val_err_hist[0].T/self.epochs
        
        for i in xrange(1,len(err_hist)):
            a = numpy.concatenate((a,err_hist[i].T/self.epochs))
            b = numpy.concatenate((b,val_err_hist[i].T/self.epochs))
        
        a = numpy.mat(a).T
        b = numpy.mat(b).T
        pylab.figure()
        for i in xrange(0,size(activities,1)):
            pylab.hold(True)
            pylab.plot(numpy.array(a[i])[0])

        pylab.figure()    
        for i in xrange(0,size(activities,1)):
            pylab.hold(True)
            pylab.plot(numpy.array(b[i])[0])

        # recalculate a new model DC based on what we have learned
        self.DC*=0.0
         
        # set weights to the minimum ones
        self.weigths = best_weights
        #for i in xrange(0,len(validation_inputs)):
        #    self.DC+=(validation_activities[i].T - self.weigths*validation_inputs[i].T).T
        #self.DC = self.DC/len(validation_inputs)   
        
	#!!!!!!!!!
	validation_error=numpy.mat(numpy.zeros(shape(activities[0].T)))
	variance=numpy.mat(numpy.zeros(shape(activities[0].T)))
	for i in xrange(0,len(validation_inputs)):
                error = ((validation_activities[i].T - self.weigths*validation_inputs[i].T - self.DC.T))
                validation_error=validation_error+numpy.power(error,2)
		variance = variance + numpy.power((validation_activities[i].T - numpy.mean(validation_activities,axis=0).T),2) 
	
	print "FINAL VALIDATION ERROR", numpy.sum(validation_error)/len(validation_inputs)/len(validation_error)
	print "FINAL FEV on validation set", numpy.mean(1-numpy.divide(validation_error,variance))   
	
        return (min_val_err,numpy.argmin(b.T,axis=0),min_val_err_array/len(validation_inputs))
    
    def returnPredictedActivities(self,inputs):
        for i in xrange(0,len(inputs)):
           if i == 0: 
               modelActivities = self.calculateModelOutput(inputs,i)
           else:
               a = self.calculateModelOutput(inputs,i)
               modelActivities = numpy.concatenate((modelActivities,a),axis=1)
           
        return numpy.mat(modelActivities).T
        
    
    def calculateReliabilities(self,inputs,activities,top_percentage):
        err=numpy.zeros(self.num_of_units)
        modelResponses=[]
        modelActivities=[]
            
        for i in xrange(0,len(inputs)):
           if i == 0: 
               modelActivities = self.calculateModelOutput(inputs,i)
           else:
               a = self.calculateModelOutput(inputs,i)
               modelActivities = numpy.concatenate((modelActivities,a),axis=1)
           
        modelActivities = numpy.mat(modelActivities)
        
        #for i in xrange(0,self.num_of_units):
        #    corr_coef[i] = numpy.corrcoef(modelActivities[i], activities.T[i])[0][1]
        #print numpy.shape(modelActivities)
        #print numpy.shape(activities)
        for i in xrange(0,self.num_of_units):
            err[i] = numpy.sum(numpy.power(modelActivities[i]- activities.T[i],2))

        t = []
        import operator
        for i in xrange(0,self.num_of_units):
            t.append((i,err[i]))
        #t=sorted(t, key=operator.itemgetter(1))
        self.reliable_indecies*=0     
        
        for i in xrange(0,self.num_of_units*top_percentage/100):   
            self.reliable_indecies[t[i][0]] = 1
            #print t[self.num_of_units-1-i][0]
            #pylab.figure()
            #pylab.show._needmain=False            
            #pylab.subplot(3,1,1)
            #pylab.plot(numpy.array(activities.T[t[self.num_of_units-1-i][0]][0].T))
            #pylab.plot(numpy.array(modelActivities[t[self.num_of_units-1-i][0]][0].T))
	    #pylab.show()

    def testModel(self,inputs,activities,target_inputs=None):
        modelActivities=[]
        modelResponses=[]
        error = 0
        
        if target_inputs == None:
           target_inputs = [a for a in xrange(0,len(inputs))]

        for index in range(len(inputs)):
            modelActivities.append(self.calculateModelOutput(inputs,index))
            
        tmp = []
        correct = 0
        for i in target_inputs:
            tmp = []
            for j in target_inputs:
                 tmp.append(numpy.sum(numpy.power(numpy.multiply(activities[i].T-modelActivities[j],numpy.mat(self.reliable_indecies).T),2)))
                 
                 #tmp.append(numpy.sum(numpy.abs(                                    numpy.multiply(activities[i].T  - modelActivities[j],numpy.mat(self.reliable_indecies).T))                                    ))
                 #tmp.append(numpy.corrcoef(modelActivities[j].T, activities[i])[0][1])
            x = numpy.argmin(array(tmp))

            #x = numpy.argmax(array(tmp))
            x = target_inputs[x]
            #print (x,i)
            
            if (i % 1) ==1:
                pylab.show._needmain=False
                pylab.figure()
                pylab.subplot(3,1,1)
                pylab.plot(numpy.array(activities[i])[0],'o',label='traget')
                pylab.plot(numpy.array(modelActivities[x].T)[0], 'o',label='predicted model')
                pylab.plot(numpy.array(modelActivities[i].T)[0], 'o',label='correct model')
                pylab.legend()
                #pylab.show()

            if x == i:
                 correct+=1.0
                
        print correct, " correct out of ", len(target_inputs)                  
        print "Percentage of correct answers:" ,correct/len(target_inputs)*100, "%"


    def testModelBiased(self,inputs,activities,t):
        modelActivities=[]
        modelResponses=[]
        error = 0

        (num_inputs,act_len)= numpy.shape(activities)
        print (num_inputs,act_len)

        for index in range(num_inputs):
            modelActivities.append(self.calculateModelOutput(inputs,index))

        m = numpy.array(numpy.mean(activities,0))[0]
        
        tmp = []
        correct = 0
        for i in xrange(0,num_inputs):
            tmp = []
            significant_neurons=numpy.zeros(numpy.shape(activities[0]))       
            for z in xrange(0,act_len):
                if activities[i,z] >= m[z]*t: significant_neurons[0,z]=1.0
            
            for j in xrange(0,num_inputs):

                 tmp.append(numpy.sum(numpy.power(numpy.multiply(numpy.multiply(activities[i].T-modelActivities[j],numpy.mat(self.reliable_indecies)),numpy.mat(significant_neurons).T),2))/ numpy.sum(significant_neurons))
            
            x = numpy.argmin(array(tmp))
            if x == i: correct+=1.0
                
        print correct, " correct out of ", num_inputs                  
        print "Percentage of correct answers:" ,correct/num_inputs*100, "%"


class MotionModelFit(ModelFit):
    
      real_time=True
          
      def init(self):
          self.reliable_indecies = ones(self.num_of_units)
          for freq in [1.0,2.0,4.0,8.0]:
             for xpos in xrange(0,int(freq)):
                for ypos in xrange(0,int(freq)):
                    x=xpos*(self.retina_diameter/freq)-self.retina_diameter/2 + self.retina_diameter/freq/2 
                    y=ypos*(self.retina_diameter/freq)-self.retina_diameter/2 + self.retina_diameter/freq/2
                    for orient in xrange(0,8):
                        g1 = []
                        g2 = []
                        t = 2
                        sigma = 1.0
                        for speed in [3,6,30]:
                            for p in xrange(0,speed):
                                #temporal_gauss = numpy.exp(-(p-(t+1))*(p-(t+1)) / 2*sigma)
                                temporal_gauss=1.0
                                g1.append(temporal_gauss*Gabor(bounds=BoundingBox(radius=self.retina_diameter/2),frequency=freq,x=x,y=y,xdensity=self.density,ydensity=self.density,size=1/freq,orientation=2*numpy.pi/8*orient,phase=p*(numpy.pi/(speed)))())
                                g2.append(temporal_gauss*Gabor(bounds=BoundingBox(radius=self.retina_diameter/2),frequency=freq,x=x,y=y,xdensity=self.density,ydensity=self.density,size=1/freq,orientation=2*numpy.pi/8*orient,phase=p*(numpy.pi/(speed))+numpy.pi/2)())
                            self.filters.append((g1,g2))  
  
      def calculateModelResponse(self,inputs,index):
            if self.real_time:
                res = []
                for (gabor1,gabor2) in self.filters:
                    r1 = 0
                    r2 = 0
                    r=0
                    l = len(gabor1)
                    for i in xrange(0,numpy.min([index+1,l])): 
                        r1 += numpy.sum(numpy.multiply(gabor1[l-1-i],inputs[index-i]))
                        r2 += numpy.sum(numpy.multiply(gabor2[l-1-i],inputs[index-i]))
                    res.append(numpy.sqrt(r1*r1+r2*r2))
                    #res.append(r)
                    #numpy.max([res.append(r1),res.append(r2)])
            else: 
                res = []
                for (gabor1,gabor2) in self.filters:
                    r1 = 0
                    r2 = 0
                    r=0
                    li = len(inputs[index])
                    l = len(gabor1)
                    for i in xrange(0,li): 
                        r1 += numpy.sum(numpy.multiply(gabor1[l-1-numpy.mod(i,l)],inputs[index][li-1-i]))
                        r2 += numpy.sum(numpy.multiply(gabor2[l-1-numpy.mod(i,l)],inputs[index][li-1-i]))
                        #r += numpy.sqrt(r1*r1+r2*r2)
                    res.append(numpy.sqrt(r1*r1+r2*r2))
                    #res.append(r)
            
            
            return numpy.mat(res)    
      
class BasicBPModelFit(ModelFit):
 
    def init(self):
          self.reliable_indecies = ones(self.num_of_units)
          import libfann
          self.ann = libfann.neural_net()
    
    def calculateModelOutput(self,inputs,index):
        import libfann
        return numpy.mat(self.ann.run(numpy.array(inputs[index].T))).T
    
    def trainModel(self,inputs,activities,validation_inputs,validation_activities):
        import libfann
        delta=[]

        connection_rate = 1.0
        num_input = len(inputs[0])
        num_neurons_hidden = numpy.size(activities,1)
        num_output = numpy.size(activities,1)
        
        print (num_input,num_neurons_hidden,num_output)
        
        desired_error = 0.000001
        max_iterations = 1000
        iterations_between_reports = 1
        self.ann.create_sparse_array(connection_rate, (num_input, num_neurons_hidden, num_output))
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
        
        train_data  = libfann.training_data()
        test_data = libfann.training_data()
        print shape(inputs)
        print shape(activities)
        train_data.set_train_dataset(numpy.array(inputs),numpy.array(activities))
        test_data.set_train_dataset(numpy.array(validation_inputs),numpy.array(validation_activities))
        
        self.ann.reset_MSE()
        self.ann.test_data(test_data)
        print "MSE error on test data: %f" % self.ann.get_MSE()
        self.ann.reset_MSE()
        self.ann.test_data(train_data)
        print "MSE error on train data: %f" % self.ann.get_MSE()
        self.ann.reset_MSE()

        for i in range(0,self.epochs):
            e = self.ann.train_epoch(train_data)
            self.ann.reset_MSE()
            self.ann.test_data(test_data)
            print "%d > MSE error on train/test data: %f / %f" % (i,e,self.ann.get_MSE())
        
        self.ann.reset_MSE()
        self.ann.test_data(test_data)
        print "MSE error on test data: %f" % self.ann.get_MSE()
        self.ann.reset_MSE()


def showMotionEnergyPatterns():

    topo.sim['Retina']=GeneratorSheet(nominal_density=24.0,
                                  input_generator=SineGrating(),
                                  period=1.0, phase=0.01,
                                  nominal_bounds=BoundingBox(radius=0.5))
    mf = MotionModelFit()
    mf.retina_diameter = 1.0
    mf.density = topo.sim["Retina"].nominal_density
    mf.init()
 
 
 
    for i in xrange(0,8):
        g1,g2 = mf.filters[i]
        pylab.figure()
        for g in g1:
            pylab.imshow(g)
            pylab.show._needmain=False
            pylab.show()



def calculateReceptiveField(RFs,weights):
    RF = numpy.zeros(shape(RFs[0][0]))
    i = 0
    for (rf1,rf2) in RFs:
        RF += weights.T[i,0]*rf1
        #RF += weights.T[i,0]*rf2
        i+=1
    return RF
              


def generate_pyramid_model(num_or,freqs,num_phase,size):
    filters=[]
    for freq in freqs:
        for orient in xrange(0,num_or):
            g1 = Gabor(bounds=BoundingBox(radius=0.5),frequency=1.0,x=0.0,y=0.0,xdensity=size/freq,ydensity=size/freq,size=0.3,orientation=numpy.pi/8*orient,phase=numpy.pi)
            g2 = Gabor(bounds=BoundingBox(radius=0.5),frequency=1.0,x=0.0,y=0.0,xdensity=size/freq,ydensity=size/freq,size=0.3,orientation=numpy.pi/8*orient,phase=0)
            filters.append((g1(),g2()))
            #for p in xrange(0,num_phase):
            #    g = Gabor(bounds=BoundingBox(radius=0.5),frequency=1.0,x=0.0,y=0.0,xdensity=size/freq,ydensity=size/freq,size=0.3,orientation=numpy.pi/8*orient,phase=p*2*numpy.pi/num_phase)
            #    filters.append(g())

    return filters

def apply_filters(inputs, filters, spacing):
    (sizex,sizey) = numpy.shape(inputs[0])
    out = []
    for i in inputs:
        o  = []
        for f in filters:
            (f1,f2) = f
            (s,tresh) = numpy.shape(f1)
            step = int(s*spacing)
            x=0
            y=0
            while (x*step + s) < sizex:
                while (y*step + s) < sizey:
                    a = numpy.sum(numpy.sum(numpy.multiply(f1,i[x*step:x*step+s,y*step:y*step+s])))
                    b = numpy.sum(numpy.sum(numpy.multiply(f2,i[x*step:x*step+s,y*step:y*step+s])))
                    o.append(numpy.sqrt(a*a+b*b))
                    y+=step 
                x+=step
        print len(o)
        out.append(numpy.array(o))
    return numpy.array(out)
                
def clump_low_responses(dataset,threshold):
    (index,data) = dataset
    
    avg=0
    count=0
    
    for cell in data:
        for stimuli in cell:
           for rep in stimuli:
                for frame in rep:
                    avg+=frame
                    count+=1
    avg = avg/count
   
    for index1, cell in enumerate(data): 
        for index2, stimulus in enumerate(cell):
            for index3, repetition in enumerate(stimulus):
                for index4, frame in enumerate(repetition):
                    if frame>=avg*(1.0+threshold):
                       repetition[index4]=frame
                    else:
                       repetition[index4]=0
                       
    return (index,data)




def runModelFit():
   
    density=__main__.__dict__.get('density', 20)
    
    #dataset = loadRandomizedDataSet("Flogl/JAN1/20090707__retinotopy_region1_stationary_testing01_1rep_125stim_ALL",6,15,125,60)
    dataset = loadSimpleDataSet("Flogl/DataNov2009/(20090925_14_36_01)-_retinotopy_region2_sequence_50cells_2700images_on_&_off_response",2700,50)

    #this dataset has images numbered from 1
    (index,data) = dataset
    index+=1
    dataset=(index,data)
     
    #print shape(dataset[1])
    #dataset = clump_low_responses(dataset,__main__.__dict__.get('ClumpMag',0.0))
    print shape(dataset[1])
    dataset = averageRangeFrames(dataset,0,1)
    print shape(dataset[1])
    dataset = averageRepetitions(dataset)
    print shape(dataset[1])
    
    (testing_data_set,dataset) = splitDataset(dataset,0.015)    
    (validation_data_set,dataset) = splitDataset(dataset,0.1)



    training_set = generateTrainingSet(dataset)
    training_inputs=generateInputs(dataset,"/home/antolikjan/topographica/topographica/Flogl/DataOct2009","/20090925_image_list_used/image_%04d.tif",density,1.8,offset=1000)
    
    validation_set = generateTrainingSet(validation_data_set)
    validation_inputs=generateInputs(validation_data_set,"/home/antolikjan/topographica/topographica/Flogl/DataOct2009","/20090925_image_list_used/image_%04d.tif",density,1.8,offset=1000)
    
    testing_set = generateTrainingSet(testing_data_set)
    testing_inputs=generateInputs(testing_data_set,"/home/antolikjan/topographica/topographica/Flogl/DataOct2009","/20090925_image_list_used/image_%04d.tif",density,1.8,offset=1000)
    

    
    #print numpy.shape(training_inputs[0])
    #compute_spike_triggered_average_rf(training_inputs,training_set,density)
    #pylab.figure()
    #pylab.imshow(training_inputs[0])
    #pylab.show()
    #return
    
    if __main__.__dict__.get('NormalizeInputs',True):
       avgRF = compute_average_input(training_inputs)
       training_inputs = normalize_image_inputs(training_inputs,avgRF)
       validation_inputs = normalize_image_inputs(validation_inputs,avgRF)
       testing_inputs = normalize_image_inputs(testing_inputs,avgRF)
    
    
    (x,y)= numpy.shape(training_inputs[0])
    training_inputs = cut_out_images_set(training_inputs,int(y*0.4),(int(x*0.1),int(y*0.4)))
    validation_inputs = cut_out_images_set(validation_inputs,int(y*0.4),(int(x*0.1),int(y*0.4)))
    testing_inputs = cut_out_images_set(testing_inputs,int(y*0.4),(int(x*0.1),int(y*0.4)))
    #training_inputs = cut_out_images_set(training_inputs,int(density*0.33),(0,int(density*0.33)))
    #validation_inputs = cut_out_images_set(validation_inputs,int(density*0.33),(0,int(density*0.33)))
    #testing_inputs = cut_out_images_set(testing_inputs,int(density*0.33),(0,int(density*0.33)))
    
    sizex,sizey=numpy.shape(training_inputs[0])
    
    print sizex,sizey
    
    if __main__.__dict__.get('Gabor',True):
        fil = generate_pyramid_model(__main__.__dict__.get('num_or',8),__main__.__dict__.get('freq',[1,2,4]),__main__.__dict__.get('num_phase',8),numpy.min(numpy.shape(training_inputs[0])))
        
        print len(fil)
        training_inputs = apply_filters(training_inputs, fil, __main__.__dict__.get('spacing',0.1))
        testing_inputs = apply_filters(testing_inputs, fil, __main__.__dict__.get('spacing',0.1))
        validation_inputs = apply_filters(validation_inputs, fil, __main__.__dict__.get('spacing',0.1))
    else:
        training_inputs = generate_raw_training_set(training_inputs)
        testing_inputs = generate_raw_training_set(testing_inputs)
        validation_inputs = generate_raw_training_set(validation_inputs)
    
    if __main__.__dict__.get('NormalizeActivities',True):
        (a,v) = compute_average_min_max(numpy.concatenate((training_set,validation_set),axis=0))
        training_set = normalize_data_set(training_set,a,v)
        validation_set = normalize_data_set(validation_set,a,v)
        testing_set = normalize_data_set(testing_set,a,v)
    
    print shape(training_set)
    print shape(training_inputs)
    
    
    #mf = BasicBPModelFit()
    
    #mf.retina_diameter = 1.2
    mf = ModelFit()
    mf.density = density
    mf.learning_rate = __main__.__dict__.get('lr',0.1)
    mf.epochs=__main__.__dict__.get('epochs',1000)
    mf.num_of_units = 50
    mf.init()
    
    pylab.hist(training_set.flatten())

    (err,stop,min_errors) = mf.trainModel(mat(training_inputs),numpy.mat(training_set),mat(validation_inputs),numpy.mat(validation_set))
    print "\nStop criterions", stop
    print "\nNon-zero stop criterions",numpy.nonzero(stop)
    print "\nMinimal errors per cell",numpy.nonzero(min_errors)
    
    print "Model test with all neurons"
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),0.1)
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),0.3)
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),0.6)
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),1.0)
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),2.0)
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),3.0)
    mf.testModelBiased(mat(testing_inputs),numpy.mat(testing_set),4.0)
    
    print "Model test with double weights"
    mf.weigths*=2.0
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))
    mf.weigths/=2.0
    
    print "Model test on validation inputs"
    mf.testModel(mat(validation_inputs),numpy.mat(validation_set))
    
    #print "Model test on training inputs"
    #mf.testModel(mat(training_inputs),mat(training_set))
    
    
    
    mf.calculateReliabilities(mat(testing_inputs),numpy.mat(testing_set),95)
    print "95: " , mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))

    mf.calculateReliabilities(mat(testing_inputs),numpy.mat(testing_set),90)
    print "90: " , mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))

    mf.calculateReliabilities(mat(testing_inputs),numpy.mat(testing_set),50)
    print "50: " , mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))
    
    mf.calculateReliabilities(mat(testing_inputs),numpy.mat(testing_set),40)
    print "40: " , mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))

    mf.calculateReliabilities(mat(testing_inputs),numpy.mat(testing_set),30)
    print "30: " , mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))

    mf.calculateReliabilities(mat(testing_inputs),numpy.mat(testing_set),20)
    print "20: " , mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))
    
    mf.reliable_indecies=(stop>=100.0)*1.0
    print mf.reliable_indecies
    mf.testModel(mat(testing_inputs),numpy.mat(testing_set))

    lookForCorrelations(mf, numpy.mat(training_set),numpy.mat(training_inputs))
    lookForCorrelations(mf, numpy.mat(validation_set),numpy.mat(validation_inputs))
    
    pylab.show()
    return (mf,mat(testing_inputs),mat(testing_set))

def showRF(mf,indexes,x,y):
    pylab.figure()
    pylab.show._needmain=False
    #pylab.subplot(9,7,1)
    print numpy.min(numpy.min(mf.weigths))
    print numpy.max(numpy.max(mf.weigths))
    for i in indexes:
        pylab.subplot(9,7,i+1)
        w = mf.weigths[i].reshape(x,y)
         
        pylab.imshow(w,vmin=numpy.min(mf.weigths[i]),vmax=numpy.max(mf.weigths[i]),cmap=pylab.cm.RdBu)
    pylab.show()
    

def analyseDataSet(data_set):
#        for cell in dataset:
        for z in xrange(0,10):
                pylab.figure()
                pylab.plot(numpy.arange(0,num_im,1),a[z],'bo')  
        pylab.show()

def set_fann_dataset(td,inputs,outputs):
    import os
    f = open("./tmp.txt",'w')
    f.write(str(len(inputs))+" "+str(size(inputs[0],1))+" "+ str(size(outputs,1)) + "\n")
    
    
    for i in range(len(inputs)):
        for j in range(size(inputs[0],1)):
            f.write(str(inputs[i][0][j]))
            f.write('\n')
        for j in range(size(outputs,1)):
            f.write(str(outputs[i,j]))
            f.write('\n')
    f.close()
    
    td.read_train_from_file("./tmp.txt")
    

def regulerized_inverse_rf(inputs,activities,sizex,sizey,alpha,validation_inputs,validation_activities,dd,display=False):
    p = len(inputs[0])
    np = len(activities[0])
    inputs = numpy.mat(inputs)
    activities = numpy.mat(activities)
    validation_inputs = numpy.mat(validation_inputs)
    validation_activities = numpy.mat(validation_activities)
    S = numpy.mat(inputs).copy()
    for x in xrange(0,sizex):
        for y in xrange(0,sizey):
            norm = numpy.mat(numpy.zeros((sizex,sizey)))
            norm[x,y]=4
            if x > 0:
               norm[x-1,y]=-1
            if x < sizex-1:
               norm[x+1,y]=-1   
            if y > 0:
               norm[x,y-1]=-1
            if y < sizey-1:
               norm[x,y+1]=-1
            S = numpy.concatenate((S,alpha*norm.flatten()),axis=0)
    
    activities_padded = numpy.concatenate((activities,numpy.mat(numpy.zeros((sizey*sizex,np)))),axis=0)
    Z = numpy.linalg.pinv(S)*activities_padded
    Z=Z.T
    ma = numpy.max(numpy.max(Z))
    mi = numpy.min(numpy.min(Z))
    m = max([abs(ma),abs(mi)])
    RFs=[]
    of = run_nonlinearity_detection(activities,inputs*Z.T,10,display)
     
    predicted_activities = inputs*Z.T
    validation_predicted_activities = validation_inputs*Z.T
    
    tf_predicted_activities = apply_output_function(predicted_activities,of)
    tf_validation_predicted_activities = apply_output_function(validation_predicted_activities,of)
    
    
    errors = numpy.sum(numpy.power(validation_activities - validation_predicted_activities,2),axis=0)
    tf_errors = numpy.sum(numpy.power(validation_activities - tf_validation_predicted_activities,2),axis=0)
    
    
    mean_mat = numpy.array(numpy.mean(validation_activities,axis=1).T)[0]
    
    
    corr_coef=[]
    corr_coef_tf=[]
    for i in xrange(0,np):
            corr_coef.append(numpy.corrcoef(validation_activities[:,i].T, validation_predicted_activities[:,i].T)[0][1])
	    corr_coef_tf.append(numpy.corrcoef(validation_activities[:,i].T, tf_validation_predicted_activities[:,i].T)[0][1])
    
    
    for i in xrange(0,np):
        RFs.append(numpy.array(Z[i]).reshape(sizex,sizey))
    
    av=[]
    for i in xrange(0,np):
        av.append(numpy.sqrt(numpy.sum(numpy.power(Z[i],2))))
	    
    if display:
        pylab.figure()
        pylab.title(str(alpha), fontsize=16)
        for i in xrange(0,np):
            pylab.subplot(10,11,i+1)
            w = numpy.array(Z[i]).reshape(sizex,sizey)
            pylab.show._needmain=False
            pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	    pylab.axis('off')
            
        pylab.figure()
        pylab.title("relationship", fontsize=16)    
        for i in xrange(0,np):
            pylab.subplot(10,11,i+1)
            pylab.plot(validation_predicted_activities.T[i],validation_activities.T[i],'ro')
            
        pylab.figure()
        pylab.title("relationship_tf", fontsize=16)    
        for i in xrange(0,np):
            pylab.subplot(10,11,i+1)
            pylab.plot(numpy.mat(tf_validation_predicted_activities).T[i],validation_activities.T[i],'ro')
        
        
        pylab.figure()
        pylab.title(str(alpha), fontsize=16)
        for i in xrange(0,np):
            pylab.subplot(10,11,i+1)
            w = numpy.array(Z[i]).reshape(sizex,sizey)
            pylab.show._needmain=False
            m = numpy.max([abs(numpy.min(numpy.min(w))),abs(numpy.max(numpy.max(w)))])
            pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
            pylab.axis('off')
    
    
    # prediction
    act_var = numpy.sum(numpy.power(validation_activities-array([mean_mat,]*np).T,2),axis=0)
    normalized_errors = 1-numpy.array(errors / act_var)[0]
    tf_normalized_errors = 1-numpy.array(tf_errors / act_var)[0]
    error = numpy.mean(errors)
    normalized_error = numpy.mean(normalized_errors)

    (rank,correct,tr) = performIdentification(validation_activities,validation_predicted_activities)
    (tf_rank,tf_correct,tf_tr) = performIdentification(validation_activities,tf_validation_predicted_activities)
    
    
    if display:
    	pylab.figure()
        pylab.hist(av)
        pylab.xlabel("rf_magnitued")
        
        pylab.figure()
        print shape(av)
        print shape(normalized_errors)
        pylab.plot(av,normalized_errors,'ro')
        pylab.xlabel("rf_magnitued")
        pylab.ylabel("normalized error")
        
        pylab.figure()
        pylab.plot(av,tf_normalized_errors,'ro')
        pylab.xlabel("rf_magnitued")
        pylab.ylabel("tf_normalized error")

    
        pylab.figure()
        pylab.hist(normalized_errors)
        pylab.xlabel("normalized_errors")
        
        pylab.figure()
        pylab.hist(tf_normalized_errors)
        pylab.xlabel("tf_normalized_errors")
    
        pylab.figure()
        pylab.hist(corr_coef)
        pylab.xlabel("Correlation coefficient")
	
	pylab.figure()
        pylab.hist(corr_coef_tf)
        pylab.xlabel("Correlation coefficient with transfer function")
    
    #saving section
    dd.add_data("ReversCorrelationRFs",RFs,force=True)
    dd.add_data("ReversCorrelationCorrectPercentage",correct*1.0 / len(validation_inputs)* 100,force=True)
    dd.add_data("ReversCorrelationTFCorrectPercentage",tf_correct*1.0 / len(validation_inputs) *100,force=True)
    dd.add_data("ReversCorrelationPredictedActivities",predicted_activities,force=True)
    dd.add_data("ReversCorrelationPredictedActivities+TF",tf_predicted_activities,force=True)
    dd.add_data("ReversCorrelationPredictedValidationActivities",validation_predicted_activities,force=True)
    dd.add_data("ReversCorrelationPredictedValidationActivities+TF",tf_validation_predicted_activities,force=True)
    dd.add_data("ReversCorrelationNormalizedErrors",normalized_errors,force=True)
    dd.add_data("ReversCorrelationNormalizedErrors+TF",tf_normalized_errors,force=True)
    dd.add_data("ReversCorrelationCorrCoefs",corr_coef,force=True)
    dd.add_data("ReversCorrelationCorrCoefs+TF",corr_coef_tf,force=True)
    dd.add_data("ReversCorrelationTransferFunction",of,force=True)
    dd.add_data("ReversCorrelationRFMagnitude",av,force=True)
    
    print "Correct:", correct ," out of ", len(validation_inputs), " percentage:", correct*1.0 / len(validation_inputs)* 100 ,"%"
    print "TFCorrect:", tf_correct, " out of ", len(validation_inputs), " percentage:", tf_correct*1.0 / len(validation_inputs) *100 ,"%"
    print "Normalized_error:", normalized_error
    return (normalized_errors,tf_normalized_errors,correct,tf_correct,RFs,predicted_activities,validation_predicted_activities,corr_coef,corr_coef_tf) 

def run_nonlinearity_detection(activities,predicted_activities,num_bins=20,display=False):
            (num_act,num_neurons) = numpy.shape(activities)
            
            os = []
            if display:
               pylab.figure()
            for i in xrange(0,num_neurons):
                min_pact = numpy.min(predicted_activities[:,i])
                max_pact = numpy.max(predicted_activities[:,i])
                bins = numpy.arange(0,num_bins+1,1)/(num_bins*1.0)*(max_pact-min_pact) + min_pact
                bins[-1]+=0.000001
                ps = numpy.zeros(num_bins)
                pss = numpy.zeros(num_bins)
                    
                for j in xrange(0,num_act):
                    bin = numpy.nonzero(bins>=predicted_activities[j,i])[0][0]-1
                    ps[bin]+=1
                    pss[bin]+=activities[j,i] 
                
		idx = numpy.nonzero(ps==0)
                ps[idx]=1.0
                tf = pss/ps
                tf[idx]=0.0
                
                if display:
                   pylab.subplot(13,13,i+1)
                   #pylab.plot(bins[0:-1],ps)
                   #pylab.plot(bins[0:-1],pss)
                   pylab.plot(bins[0:-1],tf)
                
                os.append((bins,tf))
            return os

def apply_output_function(activities,of):
    (x,y) = numpy.shape(activities)
    acts = numpy.zeros(numpy.shape(activities))
    for i in xrange(0,x):
        for j in xrange(0,y):
            (bins,tf) = of[j]
            
            if activities[i,j] >= numpy.max(bins):
                acts[i,j] = tf[-1]
            elif activities[i,j] <= numpy.min(bins):
                 acts[i,j] = tf[0]
            else:
                bin = numpy.nonzero(bins>=activities[i,j])[0][0]-1
                # do linear interpolation
                a = bins[bin]
                b = bins[bin+1]
                alpha = (activities[i,j]-a)/(b-a)
                
                if bin!=0:
                   c = (tf[bin]+tf[bin-1])/2
                else:
                   c = tf[bin]
                
                if bin!=len(tf)-1:
                   d = (tf[bin]+tf[bin+1])/2
                else:
                   d = tf[bin]
                
                acts[i,j] = c + (d-c)* alpha
    
    return acts

def fit_sigmoids_to_of(activities,predicted_activities,offset=True,display=True):
	
    (num_in,num_ne) = numpy.shape(activities)	
    from scipy import optimize
    rand =numbergen.UniformRandom(seed=513)
    if display: 	
    	pylab.figure()

    fitfunc = lambda p, x:  (offset*p[2])+p[3] / (1 + numpy.exp(-p[0]*(x-p[1]))) # Target function
    errfunc = lambda p,x, y: numpy.mean(numpy.power(fitfunc(p, x) - y,2)) # Distance to the target function
    
    params=[]
    for i in xrange(0,num_ne):
	min_err = 10e10
	best_p = 0
	for j in xrange(0,100):
		p0 = [20*rand(),10*(rand()-0.5),20*(rand()-0.5),10*rand()] 
		(p,success,c)=optimize.fmin_tnc(errfunc,p0[:],bounds=[(0,20),(-5,5),(-10,10),(0,10)],args=(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0]),approx_grad=True,messages=0)
		err  = errfunc(p,numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0])
		if err < min_err:
		   best_p = p 
	
        params.append(best_p)
	if display:
		pylab.subplot(13,13,i+1)
	        pylab.plot(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0],'go')
        	pylab.plot(numpy.array(predicted_activities[:,i].T)[0],fitfunc(best_p,numpy.array(predicted_activities[:,i].T)[0]),'bo')

    return params
    
def fit_exponential_to_of(activities,predicted_activities,offset=True,display=True):
	
    (num_in,num_ne) = numpy.shape(activities)	
    from scipy import optimize 	
    pylab.figure()

    fitfunc = lambda p, x: offset*p[0] + p[1] * numpy.exp(p[2]*(x-p[3])) # Target function
    errfunc = lambda p,x, y: numpy.mean(numpy.power(fitfunc(p, x) - y,2)) # Distance to the target function
    
    params=[]
    for i in xrange(0,num_ne):
    	p0 = [0.0,1.0,0.1,0.0] # Initial guess for the parameters
	(p,success,c)=optimize.fmin_tnc(errfunc,p0[:],bounds=[(-20,20),(-10,10),(0,10),(-5,5)],args=(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0]),approx_grad=True,messages=0)
        params.append(p)
	if display:
		pylab.subplot(13,13,i+1)
	        pylab.plot(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0],'go')
        	pylab.plot(numpy.array(predicted_activities[:,i].T)[0],fitfunc(p,numpy.array(predicted_activities[:,i].T)[0]),'bo')

    return params
    
def fit_power_to_of(activities,predicted_activities,display=True):
	
    (num_in,num_ne) = numpy.shape(activities)	
    from scipy import optimize 	
    pylab.figure()

    fitfunc = lambda p, x: p[0] + p[1] * numpy.power(x,p[2]) # Target function
    errfunc = lambda p,x, y: numpy.mean(numpy.power(fitfunc(p, x) - y,2)) # Distance to the target function
    
    params=[]
    for i in xrange(0,num_ne):
    	p0 = [0.0,1.0,-0.5] # Initial guess for the parameters
	(p,success,c)=optimize.fmin_tnc(errfunc,p0[:],bounds=[(-20,20),(-1,1),(-1,2)],args=(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0]),approx_grad=True,messages=0)
        params.append(p)
	if display:
		pylab.subplot(13,13,i+1)
	        pylab.plot(numpy.array(predicted_activities[:,i].T)[0],numpy.array(activities[:,i].T)[0],'go')
        	pylab.plot(numpy.array(predicted_activities[:,i].T)[0],fitfunc(p,numpy.array(predicted_activities[:,i].T)[0]),'bo')

    return params
    
    
    
def apply_sigmoid_output_function(activities,of,offset=True):
    sig = lambda p, x: (offset*p[2]) + p[3] * 1 / (1 + numpy.exp(-p[0]*(x-p[1])))
    (x,y) = numpy.shape(activities)	
    new_acts = numpy.zeros((x,y))
    
    for i in xrange(0,y):
	new_acts[:,i] = sig(of[i],numpy.array(activities[:,i].T)[0]).T
    return new_acts

def apply_exponential_output_function(activities,of,offset=True):
    sig = lambda p, x: offset*p[0] + p[1] * numpy.exp(p[2]*(x-p[3])) 
    (x,y) = numpy.shape(activities)	
    new_acts = numpy.zeros((x,y))
    
    for i in xrange(0,y):
	new_acts[:,i] = sig(of[i],numpy.array(activities[:,i].T)[0]).T
    return new_acts

def apply_power_output_function(activities,of):
    sig = lambda p, x: p[0] + p[1] * numpy.power(x,p[2])
    (x,y) = numpy.shape(activities)	
    new_acts = numpy.zeros((x,y))
    
    for i in xrange(0,y):
	new_acts[:,i] = sig(of[i],numpy.array(activities[:,i].T)[0]).T
    return new_acts


def later_interaction_prediction(activities,predicted_activities,validation_activities,validation_predicted_activities,raw_validation_set,node,display=True):
    
    (num_pres,num_neurons) = numpy.shape(activities)
    
    cor_orig = numpy.zeros((num_neurons,num_neurons))
    cor = numpy.zeros((num_neurons,num_neurons))
    
    residues = activities - predicted_activities
    
    for i in xrange(0,num_neurons):
        for j in xrange(0,num_neurons):
            cor[i,j] = numpy.corrcoef(numpy.array(residues[:,i].T),numpy.array(residues[:,j].T))[0][1]
    
    pylab.figure()
    pylab.imshow(cor,vmin=-0.1,vmax=0.5,interpolation='nearest')
    pylab.colorbar()
    
    for i in xrange(0,num_neurons):
        for j in xrange(0,num_neurons):
            cor_orig[i,j] = numpy.corrcoef(numpy.array(activities[:,i].T),numpy.array(activities[:,j].T))[0][1]
    
    pylab.figure()
    pylab.imshow(cor_orig,vmin=-0.1,vmax=0.5,interpolation='nearest')
    pylab.colorbar()
    
    
    mf = ModelFit()
    mf.learning_rate = __main__.__dict__.get('lr',0.005)
    mf.epochs=__main__.__dict__.get('epochs',4000)
    mf.num_of_units = num_neurons
    mf.init()
    
    #pylab.figure()
    #print "Weight shape",numpy.shape(mf.weigths)
    #pylab.imshow(numpy.array(mf.weigths),vmin=-1.0,vmax=1.0,interpolation='nearest')
    #pylab.colorbar()
    
    (err,stop,min_errors) = mf.trainModel(numpy.mat(activities[0:num_pres*0.9]),mat(predicted_activities[0:num_pres*0.9]),numpy.mat(activities[num_pres*0.9:-1]),mat(predicted_activities[num_pres*0.9:-1]))
    print "\nStop criterions", stop
    new_activities = mf.returnPredictedActivities(mat(activities))
    new_validation_activities = mf.returnPredictedActivities(mat(validation_activities))
    
    new_raw_validation_set = []
    for r in raw_validation_set:
	new_raw_validation_set.append(mf.returnPredictedActivities(mat(r)))
    
    
    
    ofs = fit_sigmoids_to_of(numpy.mat(new_activities),numpy.mat(predicted_activities))
    predicted_activities_t = apply_sigmoid_output_function(numpy.mat(predicted_activities),ofs)
    validation_predicted_activities_t = apply_sigmoid_output_function(numpy.mat(validation_predicted_activities),ofs)

    
    if display:
	pylab.figure()
	print "Weight shape",numpy.shape(mf.weigths)
    	pylab.imshow(numpy.array(mf.weigths),vmin=-numpy.max(numpy.abs(mf.weigths)),vmax=numpy.max(numpy.abs(mf.weigths)),interpolation='nearest')
	pylab.colorbar()
	#print numpy.sum(mf.weigths,axis=0)
	print numpy.sum(mf.weigths,axis=1)
    
	pylab.figure()
        pylab.title("model_relationship", fontsize=16)    
        for i in xrange(0,num_neurons):
            pylab.subplot(13,13,i+1)
            pylab.plot(numpy.mat(validation_activities).T[i],numpy.mat(validation_predicted_activities).T[i],'ro')

    
        pylab.figure()
        pylab.title("model_relationship", fontsize=16)    
        for i in xrange(0,num_neurons):
            pylab.subplot(13,13,i+1)
            pylab.plot(new_validation_activities.T[i],numpy.mat(validation_predicted_activities).T[i],'ro')


    

    (ranks,correct,pred) = performIdentification(validation_activities,validation_predicted_activities)
    print "ORIGINAL> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_activities - validation_predicted_activities,2))

    print '\n\nWithout TF'
    (ranks,correct,pred) = performIdentification(new_validation_activities,validation_predicted_activities)
    print "LATER> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(new_validation_activities - validation_predicted_activities,2))

    print '\n\nWith TF'
    (ranks,correct,pred) = performIdentification(new_validation_activities,validation_predicted_activities_t)
    print "LATER> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(new_validation_activities - validation_predicted_activities_t,2))



    raw_validation_data_set=numpy.rollaxis(numpy.array(new_raw_validation_set),2)
    signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, numpy.array(new_activities), numpy.array(new_validation_activities), predicted_activities, validation_predicted_activities)
    signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, numpy.array(new_activities), numpy.array(new_validation_activities), predicted_activities_t, validation_predicted_activities_t)
	
    print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power*(training_prediction_power>0)) , " / " , numpy.mean(validation_prediction_power*(validation_prediction_power>0))
    print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t*(training_prediction_power_t>0)) , " / " , numpy.mean(validation_prediction_power_t*(validation_prediction_power_t>0))


    node.add_data("LaterReversCorrelationPredictedActivities+TF",predicted_activities_t,force=True)
    node.add_data("LaterReversCorrelationPredictedValidationActivities+TF",validation_predicted_activities_t,force=True)
    node.add_data("LaterTrainingSet",new_activities,force=True)
    node.add_data("LaterValidationSet",new_validation_activities,force=True)
    node.add_data("LaterModel",mf,force=True)
    
    
    return (new_activities,new_validation_activities)
	
	

def runRFinference():
    d = contrib.dd.loadResults("results.dat")
    
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = sortOutLoading(d)
    
    e = []
    c = []
    b = []
    if False:
        x = 0.0
        for i in xrange(1,10):
            print i
            x = 0.003*i
    	    params={}
    	    params["alpha"] = __main__.__dict__.get('Alpha',x)
            db_node = db_node.get_child(params)
            (e1,te1,c1,tc1,RFs,pa,pva,corr_coef,corr_coef_tf) = regulerized_inverse_rf(training_inputs,training_set,sizex,sizey,x,validation_inputs,validation_set,db_node,True)
            e.append(e1)
            c.append(c1)
            b.append(x)
            #x = x*2
        pylab.figure()
        #pylab.semilogx(b,e)
        pylab.plot(b,numpy.mat(e))
        pylab.figure()
        #pylab.semilogx(b,c)
        pylab.plot(b,c)
	
	#f = open("results.dat",'wb')
    	#pickle.dump(d,f,-2)
    	#f.close()
    	return (e,c,b)
    
    params={}
    params["alpha"] = __main__.__dict__.get('Alpha',0.02)
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    if False:
        alphas=[120, 290,  50, 240, 290, 260, 120, 100, 290, 130, 290, 290, 230,170, 120, 190, 290, 100, 140, 290, 290,  60, 290, 290,  80, 210,50, 250, 170, 290, 290, 290,  60, 290, 290,  60, 260, 290, 290,290,  60,  90, 290, 120, 290,  80, 270, 120, 290, 290]
    	 
	RFs = []
        e = []
	c = []
	te = []
	tc = []
	#pa = []
	pva = []
	for i in xrange(0,len(alphas)):
	      print numpy.shape(training_set)
	      print numpy.shape(training_set[:,i:i+1])
	      (e1,te1,c1,tc1,RF,pa1,pva1,corr_coef,corr_coef_tf)   = regulerized_inverse_rf(training_inputs,training_set[:,i:i+1],sizex,sizey,alphas[i],validation_inputs,validation_set[:,i:i+1],False)
	      
	      print numpy.shape(RF)
	      RFs.append(RF[0])
	      e.append(e1)
	      c.append(c1)
     	      te.append(te1)
	      tc.append(tc1)
	      pa.append(pa1)
	      pva.append(pva1)
	 
	pylab.figure()
	pylab.hist(e)
	pylab.figure()
	pylab.hist(c)
	pylab.figure()
	pylab.hist(te)
	pylab.figure()
	pylab.hist(tc)
	      
	      
    	return (e,te,c,tc,RFs,pa,pva)
	      
    (e,te,c,tc,RFs,pa,pva,corr_coef,corr_coef_tf) = regulerized_inverse_rf(training_inputs,training_set,sizex,sizey,params["alpha"],validation_inputs,validation_set,db_node,True)
    
    pylab.figure()
    pylab.xlabel("fano factor")
    pylab.ylabel("normalized error")
    pylab.plot(ff,e,'ro')
    
    pylab.figure()
    pylab.xlabel("fano factor")
    pylab.ylabel("tf normalized error")
    pylab.plot(ff,te,'ro')
    
    pylab.figure()
    pylab.xlabel("fano factor")
    pylab.ylabel("correlation coef")
    pylab.plot(ff,corr_coef,'ro')
    
    pylab.figure()
    pylab.xlabel("fano factor")
    pylab.ylabel("correlation coef after transfer function")
    pylab.plot(ff,corr_coef_tf,'ro')
    
    contrib.dd.saveResults(d,"results.dat")
    
    pylab.show()
    return (training_set,pa,validation_set,pva)

def runRFFftInference():
    f = open("results.dat",'rb')
    import pickle
    d = pickle.load(f)
    f.close()
    
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = sortOutLoading(d)
    
    params={}
    params["FFTalpha"] = __main__.__dict__.get('Alpha',50)
    db_node1 = db_node
    db_node = db_node.get_child(params)
    
    # turn inputs into fft domain
    sx,sy = numpy.shape(training_set)
    
    new_training_inputs = numpy.zeros(numpy.shape(training_inputs))
    new_validation_inputs = numpy.zeros(numpy.shape(validation_inputs))
    
    fft_norm=numpy.zeros((sizex,sizey))
    #for i in xrange(0,sx):
    # 	fft_norm += numpy.abs(numpy.fft.fft2(numpy.reshape(training_inputs[i,:],(sizex,sizey))).flatten())
    #fft_norm/=sx
    for i in xrange(0,sizex):
	for j in xrange(0,sizex):
	    if i-1==sizex/2 and j-1==sizex/2:
	       fft_norm[i,j]=1
	    else:    
	       fft_norm[i,j]=1.0/numpy.power((i-1-sizex/2)**2+(j-1-sizey/2)**2,2)
    print fft_norm
    
    for i in xrange(0,sx):
	new_training_inputs[i,:] = numpy.divide(numpy.abs(numpy.fft.fft2(numpy.reshape(training_inputs[i,:],(sizex,sizey)))), fft_norm).flatten()
	#new_training_inputs[i,:] = numpy.fft.fft2(numpy.reshape(training_inputs[i,:],(sizex,sizey))).flatten()
    
    for i in xrange(0,50):
	new_validation_inputs[i,:] = numpy.divide(numpy.abs(numpy.fft.fft2(numpy.reshape(validation_inputs[i,:],(sizex,sizey)))), fft_norm).flatten()
	#new_validation_inputs[i,:] = numpy.fft.fft2(numpy.reshape(validation_inputs[i,:],(sizex,sizey))).flatten()
    
    print params["FFTalpha"]
    (e,te,c,tc,RFs,pa,pva,corr_coef,corr_coef_tf) = regulerized_inverse_rf(new_training_inputs,training_set,sizex,sizey,params["FFTalpha"],new_validation_inputs,validation_set,db_node,True)

    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(pa),10,display=True)
    pa_t = apply_output_function(numpy.mat(pa),ofs)
    pva_t = apply_output_function(numpy.mat(pva),ofs)


    (ranks,correct,pred) = performIdentification(validation_set,pva)
    print "Direct Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pva,2))
    
    (ranks,correct,pred) = performIdentification(validation_set,pva_t)
    print "Direct Correct+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pva_t,2))
    
    (ranks,correct,pred) = performIdentification(training_set,pa_t)
    print "Direct Correct+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(training_set - pa_t,2))
    
    f = open("results.dat",'wb')
    pickle.dump(d,f,-2)
    f.close()
    
    
    for i in xrange(0,sx):
	for j in xrange(0,sy):    
		z = numpy.multiply(numpy.reshape(new_training_inputs[i,:],(sizex,sizey)),RFs[j])
		pa[i,j] = numpy.mean(numpy.power(numpy.fft.ifft2(z),2))
    
    for i in xrange(0,50):
	for j in xrange(0,sy):    
		z = numpy.multiply(numpy.reshape(new_validation_inputs[i,:],(sizex,sizey)),RFs[j])
		pva[i,j] = numpy.mean(numpy.power(numpy.fft.ifft2(z),2))
    
    ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(pa),10,display=True)
    pa_t = apply_output_function(numpy.mat(pa),ofs)
    pva_t = apply_output_function(numpy.mat(pva),ofs)
		
		
    (ranks,correct,pred) = performIdentification(validation_set,pva)
    print "Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pva,2))
    
    (ranks,correct,pred) = performIdentification(validation_set,pva_t)
    print "Correct+TF:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pva_t,2))
    
    			
    return (training_set,pa,validation_set,pva)



def compute_spike_triggered_average_rf(inputs,activities,density):
    (num_inputs,num_activities) = shape(activities)
    RFs = [numpy.zeros(shape(inputs[0])) for j in xrange(0,num_activities)] 
    avgRF = numpy.zeros(shape(inputs[0]))
    for i in xrange(0,num_inputs):
        for j in xrange(0,num_activities):
            RFs[j] += activities[i][j]*inputs[i]

    for i in inputs:
        avgRF += i
    avgRF = avgRF/(num_inputs*1.0)
    
    activity_avg = numpy.zeros((num_activities,1))
    
    for z in xrange(0,num_activities):
        activity_avg[z] = numpy.sum(activities.T[z])
    
    activity_avg = activity_avg.T[0]
    
    pylab.figure()
    for j in xrange(0,10):
        fig = pylab.figure()
        pylab.show._needmain=False
        
        pylab.subplot(1,5,1)
        RFs[j]/= activity_avg[j]
        pylab.imshow(RFs[j],vmin=numpy.min(RFs[j]),vmax=numpy.max(RFs[j]))
        pylab.colorbar()
        
        pylab.subplot(1,5,2)
        pylab.imshow(RFs[j] - avgRF,vmin=numpy.min(RFs[j]- avgRF),vmax=numpy.max(RFs[j]- avgRF))
        pylab.colorbar()
        
        pylab.subplot(1,5,3)
        pylab.imshow(RFs[j]/avgRF,vmin=numpy.min(RFs[j]/avgRF),vmax=numpy.max(RFs[j]/avgRF))
        pylab.colorbar()

        pylab.subplot(1,5,4)
        pylab.imshow(avgRF,vmin=numpy.min(avgRF),vmax=numpy.max(avgRF))
        pylab.colorbar()
        pylab.show()
        #w = mf.weigths[j].reshape(density*1.2,density*1.2)
        #pylab.subplot(1,5,5)
        #pylab.imshow(w,vmin=numpy.min(w),vmax=numpy.max(w))
        #pylab.colorbar()                
        #

def analyze_rf_possition(w,level):
    import matplotlib
    from matplotlib.patches import Circle
    a= pylab.figure().gca()
    (sx,sy) = numpy.shape(w[0])
    
    X = numpy.tile(numpy.arange(0,sx,1),(sy,1))	
    Y = numpy.tile(numpy.arange(0,sy,1),(sx,1)).T

    cgs = []
    RFs=[]
    
    for i in xrange(0,len(w)):
            pylab.subplot(15,15,i+1)
            mi=numpy.min(numpy.min(w[i]))
            ma=numpy.max(numpy.max(w[i]))
            #z = ((w[i]<=(mi-mi*level))*1.0) * w[i] + ((w[i]>=(ma-ma*level))*1.0) * w[i]
	    z = w[i] * (numpy.abs(w[i])>= numpy.max(numpy.abs(w[i]))*(1-level))
            RFs.append(z)  
            
	    cgx = numpy.sum(numpy.multiply(X,numpy.power((abs(z)>0.0)*1.0,2)))/numpy.sum(numpy.power((abs(z)>0.0)*1.0,2))
	    cgy = numpy.sum(numpy.multiply(Y,numpy.power((abs(z)>0.0)*1.0,2)))/numpy.sum(numpy.power((abs(z)>0.0)*1.0,2))
	    
            cgs.append((cgx,cgy))
            r = numpy.max([numpy.abs(numpy.min(numpy.min(z))),numpy.abs(numpy.max(numpy.max(z)))])
            cir = Circle( (cgx,cgy), radius=1)
            pylab.gca().add_patch(cir)
            
            pylab.show._needmain=False
            pylab.imshow(z,vmin=-r,vmax=r,cmap=pylab.cm.RdBu)
    pylab.show()
    return (cgs,RFs)


def fitGabor(weights):
    from matplotlib.patches import Circle
    from scipy.optimize import leastsq,fmin,fmin_tnc,anneal
    from topo.base.arrayutil import array_argmax
    
    #(x,y) = numpy.shape(weights[0])
    #weights  = cut_out_images_set(weights,int(y*0.49),(int(x*0.1),int(y*0.4)))
    (denx,deny) = numpy.shape(weights[0])
    centers,RFs = analyze_rf_possition(weights,0.5)
    RFs=weights
    # determine frequency
    
    freqor = []
    for w in weights:
        ff = pylab.fftshift(pylab.fft2(w))
        (x,y) = array_argmax(numpy.abs(ff))
        (n,rubish) = shape(ff)
        freq = numpy.sqrt((x - n/2)*(x - n/2) + (y - n/2)*(y - n/2))
        #phase = numpy.arctan(ff[x,y].imag/ff[x,y].real)
        phase = numpy.angle(ff[x,y])
	
        if (x - n/2) != 0:
            orr = numpy.arctan((y - n/2.0)/(x - n/2.0))
        else:
            orr = numpy.pi/2
        
        if orr < 0:
           orr = orr+numpy.pi
	   
	#if phase < 0:
        #   phase = phase+2*numpy.pi
            
        freqor.append((freq,orr,phase))
    
    parameters=[]
    errors = []
    variances = []
    for j in xrange(0,len(RFs)):
	print 'nenuron',j
        minf = 0
        
        x = centers[j][0]
        y = centers[j][1] 
        
	#pylab.figure()
	#gab([x,y,0.2,freqor[j][1],freqor[j][0],freqor[j][2],1.0,0.001],weights[j]/numpy.sum(numpy.abs(weights[j])),display=True)
	
	rand =numbergen.UniformRandom(seed=513)
	
	min_x = []
	min_err = 100000000000000
	for r in xrange(0,30): 
		print 'rep',r
		x0 = [x,y,4,freqor[j][1],freqor[j][0]/denx,freqor[j][2],1.0,0.0002]
		#pylab.figure()
		#pylab.imshow(RFs[0])
		#gab(x0,RFs[0],display=True)
		#return
		x1 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
		
		x1[0] = x0[0] + 2.0*(rand()-0.5)*0.15*denx
		x1[1] = x0[1] + 2.0*(rand()-0.5)*0.15*deny
		x1[2] = rand()*0.2*denx
		x1[3] = rand()*numpy.pi#x0[3]+2.0*(rand()-0.5)*(numpy.pi/4)
		x1[4] = x0[4]*(rand()*2)
		x1[5] = rand()*numpy.pi
		x1[6] = 0.3 + rand()*3.7
		x1[7] = rand()*x0[7]
		
		(z,b,c) = fmin_tnc(gab,x1,bounds=[(x-denx*0.3,x+denx*0.3)        ,(y-deny*0.3,y+deny*0.3),(1.0,denx*0.5),(0.0,numpy.pi),(minf,freqor[j][0]/denx*3),(0,numpy.pi*2),(0.3,4.0),(0.0,0.1)],args=[weights[j]], xtol=0.0000000001,scale=[0.5,0.5,0.5,2.0,0.5,2.0,2.0,2.0],maxCGit=1000, ftol=0.0000000000001,approx_grad=True,maxfun=10000,eta=0.01,messages=0)
		e = gab(z,weights[j],display=False)
		if(e  < min_err):
		   min_err = e
		   min_x = z	
        
	#pylab.figure()
        #gab(min_x,weights[j],display=True)
	errors.append(min_err/(denx*deny))
	variances = numpy.var(weights[j])
        parameters.append(min_x)
    
    pylab.figure()
    pylab.hist(numpy.array(errors)/numpy.array(variances))
    pylab.xlabel('Fraction of unexplained variance')
    pylab.ylabel('# Cells')
    
    pylab.figure()
    (x,y,sigma,angle,f,p,ar,alpha) = tuple(parameters[0])
    pylab.imshow(gabor(frequency=f,x=x,y=y,xdensity=denx,ydensity=deny,size=sigma,orientation=angle,phase=p,ar=ar) * alpha)
    pylab.colorbar()    
        
    pylab.figure()		
    pylab.imshow(weights[0])
    pylab.colorbar()	
	
    pylab.figure()
    for i in xrange(0,len(parameters)):
            pylab.subplot(15,15,i+1)
            (x,y,sigma,angle,f,p,ar,alpha) = tuple(parameters[i])
            g = gabor(frequency=f,x=x,y=y,xdensity=denx,ydensity=deny,size=sigma,orientation=angle,phase=p,ar=ar) * alpha
            m = numpy.max([-numpy.min(g),numpy.max(g)])
            pylab.show._needmain=False
            pylab.imshow(g,vmin=-m,vmax=m,cmap=pylab.cm.RdBu)
    pylab.show()
    return parameters
    
    
def gab(z,w,display=False):
    from matplotlib.patches import Circle
    (x,y,sigma,angle,f,p,ar,alpha) = tuple(z)
    
    a = numpy.zeros(numpy.shape(w))
    (dx,dy) = numpy.shape(w)
    
    g =  gabor(frequency=f,x=x,y=y,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,ar=ar) * alpha
    
    if display:
        pylab.subplot(2,1,1)
        
        m = numpy.max([-numpy.min(g[0:dx,0:dy]),numpy.max(g[0:dx,0:dy])])
        cir = Circle( (y*dy,x*dx), radius=1)
        pylab.gca().add_patch(cir)
        pylab.imshow(g[0:dx,0:dy],vmin=-m,vmax=m,cmap=pylab.cm.RdBu)
        pylab.colorbar()
        pylab.subplot(2,1,2)
        m = numpy.max([-numpy.min(w),numpy.max(w)])
        cir = Circle( (y*dy,x*dx), radius=1)
        pylab.gca().add_patch(cir)
        pylab.imshow(w,vmin=-m,vmax=m,cmap=pylab.cm.RdBu)
        pylab.show._needmain=False
        pylab.colorbar()
        pylab.show()

    #print numpy.sum(numpy.power(g[0:dx,0:dy] - w,2))
    
    return numpy.sum(numpy.power(g[0:dx,0:dy] - w,2)) 

def gabor(frequency=1.0,x=0.0,y=0.0,xdensity=1.0,ydensity=1.0,size=1.0,orientation=1.0,phase=1.0,ar=1.0):
    X = numpy.tile(numpy.arange(0,xdensity,1),(ydensity,1))	
    Y = numpy.tile(numpy.arange(0,ydensity,1),(xdensity,1)).T
    X1 = (X-x)*numpy.cos(orientation) + (Y-y)*numpy.sin(orientation)
    Y1 = -(X-x)*numpy.sin(orientation) + (Y-y)*numpy.cos(orientation)
    
    ker =  - ((X1/numpy.sqrt(2)/(size*ar))**2 + (Y1/numpy.sqrt(2)/size)**2)
    g = numpy.exp(ker)*numpy.cos(2*numpy.pi*X1*frequency+phase)
    return g
    
def runSTC():
	
    f = open("modelfitDB2.dat",'rb')
    import pickle
    dd = pickle.load(f)
    f.close()	
    (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = sortOutLoading(dd)

    #params={}
    #params["alpha"] = __main__.__dict__.get('Alpha',50)
    #db_node = db_node.get_child(params)

    
			
    rfs = db_node.children[0].data["ReversCorrelationRFs"]
    
    a = STC(training_inputs-0.5,training_set[:,0:103],validation_inputs,validation_set,rfs)
    
    db_node.add_data("STCrfs",a,True)
    
    #return a
    pylab.figure()
    pylab.subplot(16,14,1)
    j=0
    
    m = []    
    for (ei,vv,vva,em,ep) in a:
        ind = numpy.argsort(numpy.abs(vv))
        w = numpy.array(ei[ind[len(ind)-1],:].real)
        m.append(numpy.max([-numpy.min(w),numpy.max(w)]))
    
    m = numpy.max(m)
    s = numpy.sqrt(sizey)     
    
     
    i=0
    acts=[]
    ofs=[]
    for (ei,vv,avv,em,ep) in a:
	ind = numpy.argsort(vv)
        pylab.figure()

	#if len(avv) == 0:
	#   acts.append([])	
	#   continue
		
	j=0
	act=[]
	act_val=[]
	of = []
		
	pylab.subplot(10,1,9)
	pylab.plot(numpy.sort(vv)[len(vv)-30:],'ro')
    	pylab.plot(em[len(vv)-30:])
	pylab.plot(ep[len(vv)-30:])
	
	pylab.subplot(10,1,10)
	pylab.plot(numpy.sort(vv)[0:20],'ro')
    	pylab.plot(em[0:20])
	pylab.plot(ep[0:20])
	
	for v in avv:
		w = numpy.array(ei[v,:].real).reshape(sizex,sizey)
		m = numpy.max([-numpy.min(w),numpy.max(w)])
		pylab.subplot(10,1,j)
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')
       	
		o = run_nonlinearity_detection((training_inputs*ei[v,:].T),numpy.mat(training_set[:,i]).T,10,display=False)
		of.append(o)
		(bins,tf) = o[0]    
		act.append(apply_output_function(training_inputs*ei[v,:].T,o))
		act_val.append(apply_output_function(validation_inputs*ei[v,:].T,o))
		pylab.subplot(10,1,j+1)
		pylab.plot(bins[0:-1],tf)
		j = j+2
	
	
	
	acts.append((act,act_val,of))

	#print "corr_coef =", numpy.corrcoef(act.T, training_set.T[i])[0][1]
	#print "PVE =",   1-numpy.sum(numpy.power(act.T- training_set.T[i],2)) / numpy.sum(numpy.power(numpy.mean(training_set.T[i])- training_set.T[i],2))
	#pylab.plot(numpy.array((training_inputs*ei[ind[len(ind)-1],:].real.T)),numpy.array(numpy.mat(training_set[:,i]).T),'ro')
	i = i+1

    db_node.add_data("STCact",acts,True)
    
    f = open("modelfitDB2.dat",'wb')
    import pickle
    dd = pickle.dump(dd,f)
    f.close()
    
    #pylab.show()
    return a

def STC(inputs,activities,validation_inputs,validation_activities,STA,cutoff=85,display=False):
    from scipy import linalg
    print "input size:",numpy.shape(inputs)
    t,s = numpy.shape(inputs)
    s = numpy.sqrt(s)
    
    (num_in,input_len) = numpy.shape(inputs)
    (num_in,act_len) = numpy.shape(activities)
    
    print numpy.mean(activities)
    
    tt = numpy.mat(numpy.zeros(numpy.shape(inputs[0])))
    
    SWa = []
    laa = []
    Ninva = []
    C = []
    eis = []
    
    for a in xrange(0,act_len):
	CC = numpy.mat(numpy.zeros((input_len,input_len)))
	U  = numpy.mat(numpy.zeros((input_len,input_len)))
	N  = numpy.mat(numpy.zeros((input_len,input_len)))
	Ninv  = numpy.mat(numpy.zeros((input_len,input_len)))
	
	for i in xrange(0,num_in):
		CC = CC + (numpy.mat(inputs[i,:]) - STA[a].flatten()/num_in).T * (numpy.mat(inputs[i,:]- STA[a].flatten()/num_in)) 
	CC = CC / num_in
	
	v,la = linalg.eigh(CC)
	la = numpy.mat(la)
	ind = numpy.argsort(v)
	for j in xrange(0,int(input_len*(cutoff/100.0))):
		v[ind[j]]=0.0
	
	for i in xrange(0,input_len):
		if v[i] != 0:     
			N[i,i] = 1/numpy.sqrt(v[i])
			Ninv[i,i] = numpy.sqrt(v[i])
		else: 
			N[i,i]=0.0
			Ninv[i,i] = 0.0
	
	U = la * numpy.mat(N)
	SW = numpy.matrix(inputs) * U
	SWa.append(SW)
	laa.append(la)
	Ninva.append(Ninv)
	
	if a == 0:
	        SW1 = SW*linalg.inv(la)
		F = numpy.mat(numpy.zeros((s,s)))
		for i in xrange(0,num_in):
			F += abs(pylab.fftshift(pylab.fft2(inputs[i,:].reshape(s,s)-STA[0])))
		pylab.figure()
		pylab.imshow(F.A,interpolation='nearest',cmap=pylab.cm.gray)
		
		
		F = numpy.mat(numpy.zeros((s,s)))
		for i in xrange(0,num_in):
			F += abs(pylab.fftshift(pylab.fft2(SW1[i,:].reshape(s,s))))
		pylab.figure()
		pylab.imshow(F.A,interpolation='nearest',cmap=pylab.cm.gray)

		
	#do significance testing
	vv=[]
	for r in xrange(0,50):
	    from numpy.random import shuffle
	    act = numpy.array(activities[:,a].T).copy()
	    shuffle(act)
	    C = numpy.zeros((input_len,input_len))	    	
	    for i in xrange(0,num_in):
	    	C += (numpy.mat(SWa[a][i,:]).T * numpy.mat(SWa[a][i,:])) * act[i]
	    C = C / num_in
	    v,ei = linalg.eigh(C)
	    vv.append(numpy.sort(v))
	vv = numpy.mat(vv)
	
	mean_diff = []
	for i in xrange(0,50):
	    for j in xrange(0,input_len-1):
		mean_diff.append(numpy.abs(vv[i,j]-vv[i,j+1]))
	
	diff_min = numpy.mean(mean_diff) - 15*numpy.std(mean_diff)
	diff_max = numpy.mean(mean_diff) + 15*numpy.std(mean_diff)
	
	error_minus = numpy.array(numpy.mean(vv,axis=0)-3.0*numpy.std(vv,axis=0))[0]	
	error_plus = numpy.array(numpy.mean(vv,axis=0)+3.0*numpy.std(vv,axis=0))[0]
	
	C = numpy.zeros((input_len,input_len))		
	for i in xrange(0,num_in):
   	    C += (numpy.mat(SWa[a][i,:]).T * numpy.mat(SWa[a][i,:])) * activities[i,a] 
	
	C = C / num_in
	
	if a == 0:
		pylab.figure()
		pylab.imshow(C)
	
	vv,ei = linalg.eigh(C)
	
	ind = numpy.argsort(vv)
	accepted=[]
	for i in xrange(len(vv)-30,len(vv)):
	    if (vv[ind[i]] >= error_plus[i]) or (vv[ind[i]] <= error_minus[i]):
	       accepted.append(i)
	
	accepted_vv=[]
	
	flag=False
	for i in accepted:
	    if i != 0:
	    	if (vv[ind[i]]-vv[ind[i-1]] >= diff_max):
	       		flag=True
	    if flag:
	       accepted_vv.append(ind[i])	
	    	
		
	print len(accepted_vv)
	
	ei=numpy.mat(ei).T
	ei = ei*(Ninva[a]*linalg.inv(laa[a]))
	eis.append((ei,vv,accepted_vv,error_minus,error_plus))
    
    return eis

def fitting():
    f = open("results.dat",'rb')
    import pickle
    dd = pickle.load(f)
    f.close()
    
    which = [0,1,4]
    
    for i in which:    
	node = dd.children[i].children[0]
	rfs = node.data["ReversCorrelationRFs"]
	params = fitGabor(rfs)
	node.add_data("FittedParams",params,force=True)
    
    #m = numpy.max([numpy.abs(numpy.min(rfs)),numpy.abs(numpy.max(rfs))])
    #pylab.figure()
    #for i in xrange(0,len(rfs)):
    #    pylab.subplot(15,15,i+1)
    #    w = numpy.array(rfs[i])
    #    pylab.show._needmain=False
    #    pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
    #	pylab.axis('off')
    #pylab.figure()
    
    
    
    f = open("results.dat",'wb')
    pickle.dump(dd,f,-2)
    f.close()
    
    return (params)

def tiling():
    contrib.modelfit.save_fig_directory='/home/antolikjan/Doc/reports/Sparsness/'
    f = open("results.dat",'rb')
    import pickle
    from matplotlib.patches import Circle
    dd = pickle.load(f)
    
    rand =numbergen.UniformRandom()
   
    rfs = [dd.children[0].children[0].data["ReversCorrelationRFs"],
    	   dd.children[1].children[0].data["ReversCorrelationRFs"],
	   dd.children[4].children[0].data["ReversCorrelationRFs"]]
    
    m=0
    for r in rfs:
        m = numpy.max([numpy.max([numpy.abs(numpy.min(r)),numpy.abs(numpy.max(r))]),m])
    loc = []
    
    f = file("./Mice/2009_11_04/region3_cell_locations", "r")
    loc.append([line.split() for line in f])
    f.close()
		
    f = file("./Mice/2009_11_04/region5_cell_locations", "r")
    loc.append([line.split() for line in f])
    f.close()
			
    f = file("./Mice/20090925_14_36_01/(20090925_14_36_01)-_retinotopy_region2_sequence_50cells_cell_locations.txt", "r")
    loc.append([line.split() for line in f])
    f.close()
    
    param=[]
    param.append(dd.children[0].children[0].data["FittedParams"])
    param.append(dd.children[1].children[0].data["FittedParams"])
    param.append(dd.children[4].children[0].data["FittedParams"])
    
    denx,deny=numpy.shape(rfs[0][0])
    
    view_angle = monitor_view_angle(59,20)
    degrees_per_pixel = view_angle / (2*denx) 
		
    for locations in loc:
	(a,b) = numpy.shape(locations)
	for i in xrange(0,a):
		for j in xrange(0,b):
			locations[i][j] = float(locations[i][j])
			
    loc[0] = numpy.array(loc[0])/256.0*261.0
    loc[1] = numpy.array(loc[1])/256.0*261.0
    loc[2] = numpy.array(loc[2])/256.0*230.0


    fitted_corr=[]
    fitted_rfs=[]
    fev = []
    for (j,rf) in zip(numpy.arange(0,len(rfs),1),rfs):
	q=[]
	g=[]
	v=[]
	rand_g=[]
	numpy.random.seed(1111)
	
	for i in xrange(0,len(rf)):
		(x,y,sigma,angle,f,p,ar,alpha) = tuple(param[j][i])
		(dx,dy) = numpy.shape(rfs[j][0])
		g.append(gabor(frequency=f,x=x,y=y,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,ar=ar) * alpha)
		#q.append(numpy.sum(numpy.power(rf[i].flatten()- numpy.mean(rf[i].flatten()),2)))
		q.append(numpy.mean(numpy.power(rf[i],2)))
		v.append(numpy.var(rf[i]- gabor(frequency=f,x=x,y=y,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,ar=ar) * alpha) / numpy.var(rf[i]))
	fitted_corr.append(q)
	fitted_rfs.append(g)
	fev.append(v)

    
    #dd.children[0].children[0].add_data("FittedRFs",fitted_rfs[0],force=True)	
    #dd.children[1].children[0].add_data("FittedRFs",fitted_rfs[1],force=True)
    #dd.children[4].children[0].add_data("FittedRFs",fitted_rfs[2],force=True)
    #f = open("results.dat",'wb')
    #pickle.dump(dd,f,-2)
    #f.close()		
    pylab.figure()
    pylab.hist(fev[0] + fev[1] + fev[2])
    pylab.xlabel('Fraction of un-explained variance')
		
    pylab.figure()
    ii=0
    for k in xrange(0,len(rfs)):		
	m = numpy.max([numpy.abs(numpy.min(rfs[k])),numpy.abs(numpy.max(rfs[k]))])
	for i in xrange(0,len(rfs[k])):
		pylab.subplot(15,15,ii+1)
		w = numpy.array(rfs[k][i])
		pylab.show._needmain=False
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		cir = Circle( (param[k][i][0],param[k][i][1]), radius=1,color='r')
                pylab.gca().add_patch(cir)
		xx,yy = centre_of_gravity(rfs[k][i])
		cir = Circle( (xx,yy), radius=1,color='b')
                pylab.gca().add_patch(cir)
		pylab.axis('off')
		ii+=1
   
    pylab.figure()	
    ii=0
    for k in xrange(0,len(rfs)):		
	m = numpy.max([numpy.abs(numpy.min(fitted_rfs[k])),numpy.abs(numpy.max(fitted_rfs[k]))])
	for i in xrange(0,len(fitted_rfs[k])):
		pylab.subplot(15,15,ii+1)
		w = numpy.array(fitted_rfs[k][i])
		pylab.show._needmain=False
		pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')
		ii+=1		
		
    for i in xrange(0,len(rfs)):
	#to_delete = numpy.nonzero((numpy.array(fitted_corr[i]) < 0.3*(10**-9))*1.0)[0]
	to_delete = numpy.nonzero((numpy.array(fev[i]) > 0.3)*1.0)[0]
	rfs[i] = numpy.delete(rfs[i],to_delete,axis=0)
	fitted_rfs[i] = numpy.delete(fitted_rfs[i],to_delete,axis=0)
	loc[i] = numpy.delete(numpy.array(loc[i]),to_delete,axis=0)
	param[i] = numpy.delete(numpy.array(param[i]),to_delete,axis=0)
    
    
    bb = [[],[],[]]
    rand_fitted_rfs=[]
    for a in xrange(0,3):
	for j in xrange(0,10):
		perm1 = numpy.random.permutation(len(param[a]))
		perm2 = numpy.random.permutation(len(param[a]))
		perm3 = numpy.random.permutation(len(param[a]))
		perm4 = numpy.random.permutation(len(param[a]))
		perm5 = numpy.random.permutation(len(param[a]))
		perm6 = numpy.random.permutation(len(param[a]))
		perm7 = numpy.random.permutation(len(param[a]))
		perm8 = numpy.random.permutation(len(param[a]))
		mmin = numpy.min(param[a],axis=0)
		mmax = numpy.max(param[a],axis=0)
		z = numpy.zeros(numpy.shape(rfs[a][0]))
		for i in xrange(0,len(rfs[a])):
			(x,y,sigma,angle,f,p,ar,alpha) = tuple(param[a][i])
			x = param[a][perm1[i]][0]
			y = param[a][perm2[i]][1]
			sigma = param[a][perm3[i]][2]
			angle = param[a][perm4[i]][3]
			f = param[a][perm5[i]][4]
			p = param[a][perm6[i]][5]
			ar = param[a][perm7[i]][6]
			alpha = param[a][perm8[i]][7]
			
			#x = (mmin+numpy.random.rand(8)*(mmax-mmin))[0]
			#y = (mmin+numpy.random.rand(8)*(mmax-mmin))[1]
			#sigma = (mmin+numpy.random.rand(8)*(mmax-mmin))[2]
			#angle = (mmin+numpy.random.rand(8)*(mmax-mmin))[3]
			#f = (mmin+numpy.random.rand(8)*(mmax-mmin))[4]
			#p = (mmin+numpy.random.rand(8)*(mmax-mmin))[5]
			#ar = (mmin+numpy.random.rand(8)*(mmax-mmin))[6]
			#alpha = (mmin+numpy.random.rand(8)*(mmax-mmin))[7]
		
			z = z+ gabor(frequency=f,x=x,y=y,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,ar=ar) * alpha
	        z = z / len(rfs[a])
		if j == 0:
		   rand_fitted_rfs.append(z)	
		bb[a].append(numpy.var(z))
	
	
    fitted_rfs_merged=numpy.concatenate(fitted_rfs)
    rfs_merged=numpy.concatenate(rfs)
    params_merged=numpy.concatenate(param)
    order = numpy.argsort(params_merged[:,4])

    
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))	
    m = numpy.max([numpy.abs(numpy.min(rfs_merged)),numpy.abs(numpy.max(rfs_merged))])
    for i in xrange(0,len(rfs_merged)):
	pylab.subplot(15,15,i+1)
	w = numpy.array(rfs_merged[i])
	pylab.show._needmain=False
	pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	pylab.axis('off')
    release_fig('RawRFs.pdf') 	
	
	
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))	
    m = numpy.max([numpy.abs(numpy.min(fitted_rfs_merged)),numpy.abs(numpy.max(fitted_rfs_merged))])
    for i in xrange(0,len(fitted_rfs_merged)):
	pylab.subplot(15,15,i+1)
	w = numpy.array(fitted_rfs_merged[i])
	pylab.show._needmain=False
	pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	pylab.axis('off')
    release_fig('FittedRFs.pdf')
	    
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    m = numpy.max([numpy.abs(numpy.min(rfs_merged)),numpy.abs(numpy.max(rfs_merged))])
    for i in xrange(0,len(order)):
	pylab.subplot(15,15,i+1)
	w = numpy.array(rfs_merged[order[i]])
	pylab.show._needmain=False
	pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	pylab.axis('off')
    release_fig('OrderedRFs.pdf')
    	    
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    m = numpy.max([numpy.abs(numpy.min(fitted_rfs_merged)),numpy.abs(numpy.max(fitted_rfs_merged))])
    for i in xrange(0,len(order)):
	pylab.subplot(15,15,i+1)
	w = numpy.array(fitted_rfs_merged[order[i]])
	pylab.show._needmain=False
	pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	pylab.axis('off')
    release_fig('OrderedFittedRFs.pdf')	    
	    
    # show RF coverage
    rc=[]
    for rf in rfs:
	r=[]
	for idx in xrange(0,len(rf)):
	    r.append(numpy.array(centre_of_gravity(rf[idx])))
        rc.append(numpy.array(r))

    nx=[]
    ny=[]
    for i in xrange(len(param)):
    	for j in xrange(len(param[i])):
		nx.append(param[i][j][4] * param[i][j][2]* param[i][j][6])    
	        ny.append(param[i][j][4] * param[i][j][2])	    
    
    # PLOT DIFFERENT FITTED PARAMETERS HISTOGRAMS
    pylab.figure(dpi=300,facecolor='w',figsize=(6,4))
    pylab.scatter(nx,ny,s=10,facecolor='none', edgecolor='b',marker='o')
    pylab.axis([0,1.5,0.0,1.5])
    pylab.axes().set_aspect('equal')
    pylab.xlabel('nx')
    pylab.ylabel('ny')
    pylab.gca().xaxis.set_major_locator(MaxNLocator(3))
    pylab.gca().yaxis.set_major_locator(MaxNLocator(3))
    release_fig('NxNy.png')

    pylab.figure(dpi=100,facecolor='w',figsize=(17,5))
    pylab.subplot(1,5,1)
    pylab.hist(numpy.array(param[0][:,4].flatten().tolist() + param[1][:,4].flatten().tolist() + param[2][:,4].flatten().tolist())*degrees_per_pixel)
    pylab.xlabel('Frequency (deg. / visual angle)')
    pylab.gca().xaxis.set_major_locator(MaxNLocator(5))

    
    pylab.subplot(1,5,2)
    pylab.hist(numpy.array(param[0][:,2].flatten().tolist() + param[1][:,2].flatten().tolist() + param[2][:,2].flatten().tolist())*degrees_per_pixel)
    pylab.xlabel('Sigma (deg. / visual angle)')
    pylab.gca().xaxis.set_major_locator(MaxNLocator(5))
    
    pylab.subplot(1,5,3)
    pylab.hist(param[0][:,6].flatten().tolist() + param[1][:,6].flatten().tolist() + param[2][:,6].flatten().tolist())
    pylab.xlabel('Aspect ratio')
    pylab.gca().xaxis.set_major_locator(MaxNLocator(5))
    
    pylab.subplot(1,5,4)
    c=[]
    for j in xrange(0,len(param)):
    	for i in xrange(0,len(param[j][:,5])):
		c.append(numpy.complex(numpy.abs(numpy.cos(param[j][i,5])),numpy.abs(numpy.sin(param[j][i,5]))))
    pylab.hist(numpy.angle(c))
    pylab.gca().xaxis.set_major_locator(MaxNLocator(5))			
    pylab.xlabel('Phase')
    
    pylab.subplot(1,5,5)
    pylab.hist(param[0][:,3].flatten().tolist() + param[1][:,3].flatten().tolist() + param[2][:,3].flatten().tolist())
    pylab.xlabel('Orientation')
    pylab.gca().xaxis.set_major_locator(MaxNLocator(5))		
    release_fig('FittedParametersDistribution.pdf')
    	
			
    #PLOT THE RETINOTOPIC COVERAGE 			
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    for i in xrange(0,len(param)):
	pylab.subplot(1,3,i+1)
	pylab.title('Retinotopic coverage')
	pylab.plot(param[i][:,0],param[i][:,1],'bo',label='Fitted')
	pylab.plot(rc[i][:,0],rc[i][:,1],'ro',label='Center of gravity')
        pylab.axis([0,numpy.shape(rfs[0][0])[0],0.0,numpy.shape(rfs[0][0])[1]])
        pylab.gca().set_aspect('equal')
	pylab.xlabel('X coordinate')
	pylab.ylabel('Y coordinate')
	pylab.legend()
    release_fig('RetinotopicCoverage.pdf')


    #PLOT THE ORIENTATION PREFERENCE AGIANST RETINOTOPY 			
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    for i in xrange(0,len(param)):
	pylab.subplot(1,3,i+1)
	pylab.title('Retinotopic coverage')
	pylab.scatter(param[i][:,0],param[i][:,1],c=param[i][:,3]/numpy.pi,s=50,cmap=pylab.cm.hsv)
        #pylab.axis([0,numpy.shape(rfs[0][0])[0],0.0,numpy.shape(rfs[0][0])[1]])
        pylab.gca().set_aspect('equal')
	pylab.xlabel('X coordinate')
	pylab.ylabel('Y coordinate')
	pylab.colorbar(shrink=0.3)
    release_fig('ORandRetinotopy.pdf')
    
    aaa = []
    d = []	
    for i in xrange(0,len(fitted_rfs[0])):
	for j in xrange(i+1,len(fitted_rfs[0])):
	    d.append(distance(loc[0],i,j))
	    aaa.append(numpy.corrcoef(fitted_rfs[0][i].flatten(),fitted_rfs[0][j].flatten())[0][1])

    pylab.figure(facecolor='w')
    pylab.title('Correlation between distance and fitted RFs correlations')
    ax = pylab.axes() 
    ax.plot(d,aaa,'ro')
    ax.plot(d,contrib.jacommands.weighted_local_average(d,aaa,30),'go')
    ax.plot(d,numpy.array(contrib.jacommands.weighted_local_average(d,aaa,30))+numpy.array(contrib.jacommands.weighted_local_std(d,aaa,30)),'bo')
    ax.plot(d,numpy.array(contrib.jacommands.weighted_local_average(d,aaa,30))-numpy.array(contrib.jacommands.weighted_local_std(d,aaa,30)),'bo')
    ax.axhline(0,linewidth=4)
    
    
    #PLOT RF COVERAGE
    #first raw	
    z = numpy.zeros(numpy.shape(rfs[0][0]))
    for f in rfs[0]:
	z = z+f
    z = z/len(rfs[0])	
    pylab.figure(dpi=100,facecolor='w',figsize=(15,11))
    pylab.subplot(1,5,1)
    pylab.title('Raw')
    m = numpy.max([numpy.abs(numpy.min(z)),numpy.abs(numpy.max(z))])
    pylab.imshow(z,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
    pylab.colorbar(shrink=0.3)
    print 'Variance of the raw averaged RFs',numpy.var(z)
    
    #then fitted
    z = numpy.zeros(numpy.shape(fitted_rfs[0][0]))
    for f in fitted_rfs[0]:
	z = z+f
    z = z/len(fitted_rfs[0])
    pylab.subplot(1,5,2)
    pylab.title('Fitted')
    m = numpy.max([numpy.abs(numpy.min(z)),numpy.abs(numpy.max(z))])
    pylab.imshow(z,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
    pylab.colorbar(shrink=0.3)
    print 'Variance of the fitted averaged RFs',numpy.var(z)
    
    #then randomized fitted
    pylab.subplot(1,5,3)
    pylab.title('Randmozied fitted')
    m = numpy.max([numpy.abs(numpy.min(rand_fitted_rfs[0])),numpy.abs(numpy.max(rand_fitted_rfs[0]))])
    pylab.imshow(numpy.array(rand_fitted_rfs[0]),vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
    pylab.colorbar(shrink=0.3)	
    
    pylab.subplot(1,5,4)
    pylab.title('Example fitted')
    m = numpy.max([numpy.abs(numpy.min(fitted_rfs[0][0])),numpy.abs(numpy.max(fitted_rfs[0][0]))])
    pylab.imshow(fitted_rfs[0][0],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
    pylab.colorbar(shrink=0.3)

    pylab.subplot(1,5,5)
    pylab.title('Example raw')
    m = numpy.max([numpy.abs(numpy.min(rfs[0][0])),numpy.abs(numpy.max(rfs[0][0]))])
    pylab.imshow(rfs[0][0],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
    pylab.colorbar(shrink=0.3)
    release_fig('ONOFFcoverage.pdf')	

    print 'Average variance and +/- 2*variance of the randomized fitted averaged RF',numpy.mean(bb[0]), numpy.mean(bb[0]) + 2*numpy.sqrt(numpy.var(bb[0])), numpy.mean(bb[0]) - 2*numpy.sqrt(numpy.var(bb[0])) 

    pylab.figure()
    pylab.subplot(1,3,1)
    pylab.hist(bb[0],bins=30)
    pylab.axvline(numpy.var(numpy.mean(fitted_rfs[0],axis=0)))
    print numpy.var(numpy.mean(fitted_rfs[0],axis=0))
    
    pylab.subplot(1,3,2)
    pylab.hist(bb[1],bins=30)
    pylab.axvline(numpy.var(numpy.mean(fitted_rfs[1],axis=0)))
    print numpy.var(numpy.mean(fitted_rfs[1],axis=0))

    pylab.subplot(1,3,3)
    pylab.hist(bb[2],bins=30)
    pylab.axvline(numpy.var(numpy.mean(fitted_rfs[2],axis=0)))
    print numpy.var(numpy.mean(fitted_rfs[2],axis=0))
    
    return

    membership=[]
    membership1=[]
    locs = []
    for i in xrange(0,len(rfs)):
	m = []
	m1 = []
	l = []
	for j in xrange(0,len(rfs[i])):
    	    if (circular_distance(param[i][j][3],0) <= numpy.pi/8):
		m.append(0)
		m1.append(0)
		l.append(loc[i][j])     	
    	    elif (circular_distance(param[i][j][3],numpy.pi/4) <= numpy.pi/8):
	    	m1.append(1)     	
    	    elif (circular_distance(param[i][j][3],numpy.pi/2) <= numpy.pi/8):
		m.append(2)
		m1.append(2)
		l.append(loc[i][j])     	
    	    elif (circular_distance(param[i][j][3],3*numpy.pi/4) <= numpy.pi/8):
	    	m1.append(3)
	         	
	membership.append(m)
	membership1.append(m1)
	locs.append(l)
				
    ors=[]
    orrf=[]		
    orrphase=[]		
    for (p,rf) in zip(param,rfs):
	ors.append(numpy.array(p[:,3].T)) 				
        orrf.append(zip(numpy.array(p[:,3].T),rf))
	orrphase.append(zip(numpy.array(p[:,3].T),numpy.array(p[:,5].T)))
	
    #monte_carlo(loc,orrphase,histogram_of_phase_dist_correl_of_cooriented_neurons,30)  
    #monte_carlo(loc,orrf,histogram_of_RF_correl_of_cooriented_neurons,30)
    #monte_carlo(loc,ors,average_or_histogram_of_proximite,30)
    #monte_carlo(loc,ors,average_or_diff,30)	
	
    #monte_carlo(loc,ors,average_or_diff,30)	
    monte_carlo(loc,orrf,average_cooriented_RF_corr,30)
    #monte_carlo(loc,orrf,average_RF_corr,30)
    return 
    
    #return
    
    #monte_carlo(loc,orrphase,histogram_of_phase_dist_correl_of_cooriented_neurons,30)  
    #monte_carlo(loc,orrf,histogram_of_RF_correl_of_cooriented_neurons,30)
    #monte_carlo(loc,ors,average_or_histogram_of_proximite,30)
    #return
    #monte_carlo(locs,membership,number_of_same_neighbours,100)				
    #monte_carlo(loc,membership1,number_of_same_neighbours,100)
        		    
    
    #return
    
    colors=[]
    xx=[]
    yy=[]
    d1=[]
    d2=[]
    d3=[]

    for r,l in zip(rfs,loc):
    	for i in xrange(0,len(r)):
	    for j in xrange(i+1,len(r)):
		d3.append(distance(l,i,j))
		
    for i in xrange(0,len(rfs)):
	colors=[]
	xx=[]
    	yy=[]
	for idx in xrange(0,len(rfs[i])):
		if (circular_distance(param[i][idx][3],0) <= numpy.pi/12):
			xx.append(loc[i][idx][0])
			yy.append(loc[i][idx][1])
			colors.append(0.9)
	    		for j in xrange(idx+1,len(rfs[i])):
			    if (circular_distance(param[i][j][3],0) <= numpy.pi/12):
			        d1.append(distance(loc[i],idx,j))	    	
				d2.append(distance(loc[i],idx,j))					    	
			    if (circular_distance(param[i][j][3],numpy.pi/2) <= numpy.pi/12):
				d2.append(distance(loc[i],idx,j))


		if (circular_distance(param[i][idx][3],numpy.pi/2) <= numpy.pi/12): 
			xx.append(loc[i][idx][0])
			yy.append(loc[i][idx][1])
			colors.append(0.1)
			for j in xrange(idx+1,len(rfs[i])):
			    if (circular_distance(param[i][j][3],numpy.pi/2) <= numpy.pi/12):
			        d1.append(distance(loc[i],idx,j))
				d2.append(distance(loc[i],idx,j))					    	
			    if (circular_distance(param[i][j][3],0) <= numpy.pi/12):
				d2.append(distance(loc[i],idx,j))
			
	pylab.figure(figsize=(5,5))
	pylab.scatter(xx,yy,c=colors,s=200,cmap=pylab.cm.RdBu)
	pylab.colorbar()
    print "Average distance of colinear", numpy.mean(numpy.power(d1,2))
    print "Average distance of horizontal and vertical", numpy.mean(numpy.power(d2,2))
    print "Average distance of whole population", numpy.mean(numpy.power(d3,2))


def monte_carlo(locations,property,property_measure,reps):
    from numpy.random import shuffle	
    for (l,m) in zip(locations,property):
    	a = property_measure(l,m)
	curves = numpy.zeros((reps,len(a)))
	
	for x in xrange(0,reps):
		mm = list(m) 
		shuffle(mm)
		curves[x,:] = property_measure(l,mm)
		
	f = numpy.median(curves,axis=0)
	f_m = a
	#std = numpy.std(curves,axis=0,ddof=1)
	err_bar_upper = numpy.sort(curves,axis=0)[int(reps*0.95),:]
	err_bar_lower = numpy.sort(curves,axis=0)[int(reps*0.05),:]
	
	pylab.figure()
	pylab.plot(f,'b')
	pylab.plot(f_m,'g')
	pylab.plot(err_bar_lower,'r')
	pylab.plot(err_bar_upper,'r')
	
def number_of_same_neighbours(locations,membership):	
    curve = [0 for i in xrange(0,30)]
    for dist in xrange(0,30):
	for i in xrange(0,len(locations)):
		for j in xrange(0,len(locations)):
			if i!=j:
				if distance(locations,i,j) < (dist+1)*10:
					if membership[i] == membership[j]:
						curve[dist] += 1
	curve[dist]/= len(locations)	
    return curve
						
def average_or_diff(locations,ors):
    curve = [0 for i in xrange(0,30)]
    for dist in xrange(0,30):
	n = 0    
	for i in xrange(0,len(locations)):
		for j in xrange(0,len(locations)):
			if i!=j:
				if distance(locations,i,j) < (dist+1)*10:
					curve[dist]+=circular_distance(ors[i],ors[j])
					n+=1
	if n!=0:				
		curve[dist]/=n	
    return curve

def average_cooriented_RF_corr(locations,data):
    (ors,rfs) = zip(*data)
    curve = [0 for i in xrange(0,30)]
    for dist in xrange(0,30):
	n = 0    
	for i in xrange(0,len(locations)):
		for j in xrange(0,len(locations)):
			if i!=j:
				if distance(locations,i,j) < (dist+1)*10:
					if circular_distance(ors[i],ors[j]) < (numpy.pi/12.0):
						curve[dist]+=numpy.corrcoef(rfs[i].flatten(),rfs[j].flatten())[0][1]
						n+=1
	if n!=0:				
		curve[dist]/=n	
    return curve
	
def average_RF_corr(locations,data):
    (ors,rfs) = zip(*data)
    curve = [0 for i in xrange(0,30)]
    for dist in xrange(0,30):
	n = 0    
	for i in xrange(0,len(locations)):
		for j in xrange(0,len(locations)):
			if i!=j:
				if distance(locations,i,j) < (dist+1)*10:
						curve[dist]+=numpy.corrcoef(rfs[i].flatten(),rfs[j].flatten())[0][1]
						n+=1
	if n!=0:				
		curve[dist]/=n	
    return curve

	
def average_or_histogram_of_proximite(locations,ors):
    curve = []
    for i in xrange(0,len(locations)):
	for j in xrange(0,len(locations)):
		if i!=j:
			if distance(locations,i,j) < 50:
				curve.append(circular_distance(ors[i],ors[j]))
    if len(curve) != 0:					
    	return numpy.histogram(curve,range=(0.0,numpy.pi/2))[0]/(len(curve)*1.0)
    else: 
    	return [0 for i in xrange(0,10)]

def histogram_of_RF_correl_of_cooriented_neurons(locations,data):
    (ors,rfs) = zip(*data)
    difs=[]
    for i in xrange(0,len(locations)):
	for j in xrange(0,len(locations)):
	    if i!=j:
		if circular_distance(ors[i],ors[j]) < (numpy.pi/8.0):
			if distance(locations,i,j) < 50:
				difs.append(numpy.corrcoef(rfs[i].flatten(),rfs[j].flatten())[0][1])
    if len(difs) != 0:						    
    	return numpy.histogram(difs,range=(-1.0,1.0),bins=10)[0]/(len(difs)*1.0)
    else:
	return [0 for i in xrange(0,10)]

def histogram_of_phase_dist_correl_of_cooriented_neurons(locations,data):
    (ors,phase) = zip(*data)
    difs=[]
    for i in xrange(0,len(locations)):
	for j in xrange(0,len(locations)):
	    if i!=j:
		if circular_distance(ors[i],ors[j]) < (numpy.pi/12.0):
			if distance(locations,i,j) < 40:
				dif = numpy.abs(phase[i] - phase[j])
				if dif > numpy.pi:
				   dif = 2*numpy.pi - dif	
				difs.append(dif)
    if len(difs) != 0:						    
    	return numpy.histogram(difs,range=(0,numpy.pi),bins=10)[0]/(len(difs)*1.0)
    else:
	return [0 for i in xrange(0,10)]


def RF_correlations():
    f = open("modelfitDB2.dat",'rb')
    import pickle
    dd = pickle.load(f)

   
    rfs = [dd.children[0].children[0].data["ReversCorrelationRFs"],
    	   dd.children[1].children[0].data["ReversCorrelationRFs"],
	   dd.children[3].children[0].data["ReversCorrelationRFs"]]
    
    m=0
    for r in rfs:
        m = numpy.max([numpy.max([numpy.abs(numpy.min(r)),numpy.abs(numpy.max(r))]),m])
    loc = []
    
    f = file("./Mice/2009_11_04/region3_cell_locations", "r")
    loc.append([line.split() for line in f])
    f.close()
		
    f = file("./Mice/2009_11_04/region5_cell_locations", "r")
    loc.append([line.split() for line in f])
    f.close()
			
    f = file("./Mice/20090925_14_36_01/(20090925_14_36_01)-_retinotopy_region2_sequence_50cells_cell_locations.txt", "r")
    loc.append([line.split() for line in f])
    f.close()

			
		
    param=[]
    
    f = open("./Mice/2009_11_04/region=3_fitting_rep=100","rb")
    import pickle
    param.append(pickle.load(f))
    f.close()
    f = open("./Mice/2009_11_04/region=5_fitting_rep=100","rb")
    param.append(pickle.load(f))
    f.close()    		
    f = open("./Mice/20090925_14_36_01/region=2_fitting_rep=100","rb")
    param.append(pickle.load(f))
    f.close()    		
		
		
    for locations in loc:
	(a,b) = numpy.shape(locations)
	for i in xrange(0,a):
		for j in xrange(0,b):
			locations[i][j] = float(locations[i][j])
			
    loc[0] = numpy.array(loc[0])/256.0*261.0
    loc[1] = numpy.array(loc[1])/256.0*261.0
    loc[2] = numpy.array(loc[2])/256.0*230.0
    
    fitted_corr=[]
    for rf in rfs:
	f=[]	    	     
	for i in xrange(0,len(rf)):
		#(x,y,sigma,angle,f,p,ar,alpha) = tuple(params[i])
		#(dx,dy) = numpy.shape(rfs[0])
		#g = Gabor(bounds=BoundingBox(radius=0.5),frequency=f,x=y-0.5,y=0.5-x,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,aspect_ratio=ar)() * alpha    
		f.append(numpy.sum(numpy.power(rf[i].flatten()- numpy.mean(rf[i].flatten()),2)))
	fitted_corr.append(f)
	
    pylab.title("The histogram of the variability of RFs")
    pylab.hist(flatten(fitted_corr))
    pylab.xlabel('RF variability')
    
    
    for i in xrange(0,len(fitted_corr)):
	pylab.figure()
	z = numpy.argsort(fitted_corr[i])
	b=0	    
	for j in z:	    
		pylab.subplot(15,15,b+1)
		pylab.show._needmain=False
		pylab.imshow(rfs[i][j],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')	
		b+=1
 
    for i in xrange(0,len(rfs)):
	to_delete = numpy.nonzero((numpy.array(fitted_corr[i]) < 0.00000004)*1.0)[0]
	rfs[i] = numpy.delete(rfs[i],to_delete,axis=0)
	loc[i] = numpy.delete(numpy.array(loc[i]),to_delete,axis=0)
	param[i] = numpy.delete(numpy.array(param[i]),to_delete,axis=0)

    rc=[]
    for rf in rfs:
	r=[]
	for idx in xrange(0,len(rf)):
	    r.append(numpy.array(centre_of_gravity(numpy.power(rf[idx],2)))*1000)
        rc.append(r)

    for i in xrange(0,len(rfs)):
	pylab.figure()
	b=0	    
	for j in rfs[i]:	    
		pylab.subplot(15,15,b+1)
		pylab.show._needmain=False
		pylab.imshow(j,vmin=-m*0.5,vmax=m*0.5,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')	
		b+=1
	pylab.savefig('RFsGrid'+str(i)+'.png')	


    rf_cross=[]
    for r in rfs:
	print len(r)
	rf_cros = numpy.zeros((len(r),len(r)))	    
	for i in xrange(0,len(rfs)):
		for j in xrange(0,len(rfs)):
			rf_cros[i,j] = numpy.corrcoef(r[i].flatten(),r[j].flatten())[0][1]
	rf_cross.append(rf_cros)
    i=0		
    for (r,locations) in zip(rfs,loc):
        pylab.figure(figsize=(5,5))
	pylab.axes([0.0,0.0,1.0,1.0])
    	for idx in xrange(0,len(r)):
		x = locations[idx][0]/300
		y = locations[idx][1]/300
		pylab.axes([x-0.02,y-0.02,0.04,0.04])
		pylab.imshow(r[idx],vmin=-m*0.5,vmax=m*0.5,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')
    	pylab.savefig('RFsLocalized'+str(i)+'.png')
	i+=1
  
    
    
    from matplotlib.lines import Line2D   
    from matplotlib.patches import Circle
    i=0
    for (r,locations,p) in zip(rfs,loc,param):
	pylab.figure(figsize=(5,5))
    	pylab.axes([0.0,0.0,1.0,1.0])
	
    	for idx in xrange(0,len(r)):
		x = locations[idx][0]/300
		y = locations[idx][1]/300
		pylab.axes([x-0.02,y-0.02,0.04,0.04])
		pylab.imshow(r[idx],vmin=-m*0.5,vmax=m*0.5,interpolation='nearest',cmap=pylab.cm.RdBu)
		pylab.axis('off')
		ax = pylab.axes([0.0,0.0,1.0,1.0])
		cir = Circle( (x,y), radius=0.01)
		pylab.gca().add_patch(cir)
		l = Line2D([x-numpy.cos(p[idx][3])*0.03,x+numpy.cos(p[idx][3])*0.03],[y-numpy.sin(p[idx][3])*0.03,y+numpy.sin(p[idx][3])*0.03],transform=ax.transAxes,linewidth=5.1, color='g')
		pylab.gca().add_line(l)
	pylab.savefig('RFsLocalizedOR'+str(i)+'.png')
	i+=1

    for (r,locations,p) in zip(rfs,loc,param):
	pylab.figure(figsize=(5,5))
    	pylab.axes([0.0,0.0,1,1])
    	for idx in xrange(0,len(r)):
		if circular_distance(p[idx][3],0)<= numpy.pi/8:    
			x = locations[idx][0]/300
			y = locations[idx][1]/300
			pylab.axes([x-0.02,y-0.02,0.04,0.04])
			pylab.imshow(r[idx],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
			pylab.axis('off')
			ax = pylab.axes([0.0,0.0,1,1])
			cir = Circle( (x,y), radius=0.01)
			pylab.gca().add_patch(cir)
			l = Line2D([x-numpy.cos(p[idx][3])*0.03,x+numpy.cos(p[idx][3])*0.03],[y-numpy.sin(p[idx][3])*0.03,y+numpy.sin(p[idx][3])*0.03],transform=ax.transAxes,linewidth=5.1, color='g')
			pylab.gca().add_line(l)

	pylab.figure(figsize=(5,5))
    	pylab.axes([0.0,0.0,1,1])
    	for idx in xrange(0,len(r)):
		if circular_distance(p[idx][3],numpy.pi/4)<= numpy.pi/8:    
			x = locations[idx][0]/300
			y = locations[idx][1]/300
			pylab.axes([x-0.02,y-0.02,0.04,0.04])
			pylab.imshow(r[idx],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
			pylab.axis('off')
			ax = pylab.axes([0.0,0.0,1,1])
			cir = Circle( (x,y), radius=0.01)
			pylab.gca().add_patch(cir)
			l = Line2D([x-numpy.cos(p[idx][3])*0.03,x+numpy.cos(p[idx][3])*0.03],[y-numpy.sin(p[idx][3])*0.03,y+numpy.sin(p[idx][3])*0.03],transform=ax.transAxes,linewidth=5.1, color='g')
			pylab.gca().add_line(l)


	pylab.figure(figsize=(5,5))
    	pylab.axes([0.0,0.0,1,1])
    	for idx in xrange(0,len(r)):
		if circular_distance(p[idx][3],numpy.pi/2)<= numpy.pi/8:    
			x = locations[idx][0]/300
			y = locations[idx][1]/300
			pylab.axes([x-0.02,y-0.02,0.04,0.04])
			pylab.imshow(r[idx],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
			pylab.axis('off')
			ax = pylab.axes([0.0,0.0,1,1])
			cir = Circle( (x,y), radius=0.01)
			pylab.gca().add_patch(cir)
			l = Line2D([x-numpy.cos(p[idx][3])*0.03,x+numpy.cos(p[idx][3])*0.03],[y-numpy.sin(p[idx][3])*0.03,y+numpy.sin(p[idx][3])*0.03],transform=ax.transAxes,linewidth=5.1, color='g')
			pylab.gca().add_line(l)

	pylab.figure(figsize=(5,5))
    	pylab.axes([0.0,0.0,1,1])
    	for idx in xrange(0,len(r)):
		if circular_distance(p[idx][3],3*numpy.pi/4)<= numpy.pi/8:    
			x = locations[idx][0]/300
			y = locations[idx][1]/300
			pylab.axes([x-0.02,y-0.02,0.04,0.04])
			pylab.imshow(r[idx],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
			pylab.axis('off')
			ax = pylab.axes([0.0,0.0,1,1])
			cir = Circle( (x,y), radius=0.01)
			pylab.gca().add_patch(cir)
			l = Line2D([x-numpy.cos(p[idx][3])*0.03,x+numpy.cos(p[idx][3])*0.03],[y-numpy.sin(p[idx][3])*0.03,y+numpy.sin(p[idx][3])*0.03],transform=ax.transAxes,linewidth=5.1, color='g')
			pylab.gca().add_line(l)


    
    for i in xrange(0,len(rfs)):
	colors=[]
	xx=[]
    	yy=[]
	for idx in xrange(0,len(rfs[i])):
		xx.append(loc[i][idx][0])
		yy.append(loc[i][idx][1])
		colors.append(param[i][idx][3]/numpy.pi)
    
	pylab.figure(figsize=(5,5))
	pylab.scatter(xx,yy,c=colors,s=200,cmap=pylab.cm.hsv)
	pylab.colorbar()
	
    for i in xrange(0,len(rfs)):
	colors=[]
	xx=[]
    	yy=[]
	for idx in xrange(0,len(rfs[i])):
		xx.append(rc[i][idx][0])
		yy.append(rc[i][idx][1])
		colors.append(param[i][idx][3]/numpy.pi)
    
	pylab.figure(figsize=(5,5))
	pylab.scatter(xx,yy,c=colors,s=200,cmap=pylab.cm.hsv)
	pylab.colorbar()
	
	
	
    c = []
    rf_dist = []
    c_cut = []
    d = []
    
    orr_diff = []
    phase_diff_of_colinear20 = []
    phase_diff_of_colinear30 = []
    phase_diff_of_colinear50 = []
    phase_diff_of_colinear100 = []
    phase_diff_of_colinear1000 = []
    for (r,locations,params) in zip(rfs,loc,param):
	for i in xrange(0,len(r)):
		for j in xrange(i+1,len(r)):
			corr1= numpy.corrcoef(r[i].flatten(),r[j].flatten())[0][1]	
			c.append(corr1)
			rf_dist.append(numpy.mean(numpy.power(r[i].flatten()-r[j].flatten(),2)))
			c_cut.append(RF_corr_centered(r[i],r[j],0.3,display=False))
			dist = distance(locations,i,j)
			d.append(dist)
			a = circular_distance(params[i][3],params[j][3])
				
			if a < numpy.pi/16:
				pd = numpy.abs(params[i][5] - params[j][5])
				if pd > numpy.pi:
					pd = 2*numpy.pi - pd
				
				if dist <= 20:     
					phase_diff_of_colinear20.append(pd)  
				if dist <= 40:     
					phase_diff_of_colinear30.append(pd)  
				if dist <= 60:     
					phase_diff_of_colinear50.append(pd)  
				if dist <= 100:     
					phase_diff_of_colinear100.append(pd)  
				if dist <= 300:     
					phase_diff_of_colinear1000.append(pd)	
			orr_diff.append(numpy.abs(circular_distance(params[i][3],params[j][3])))
    
    print len(d)
    print len(c)
  
    nn_orr_diff=[]
    nn_corr=[]
    nn_rf_dist=[]
    
    all_orr_diff=[]
    all_corr=[]
    all_rf_dist=[]
    for (r,locations,params) in zip(rfs,loc,param):
	for i in xrange(0,len(r)):
		dst = []
		for j in xrange(0,len(r)):
			dst.append(distance(locations,i,j))
			if i != j:
			   all_orr_diff.append(circular_distance(params[i][3],params[j][3]))
			   all_corr.append(numpy.corrcoef(r[i].flatten(),r[j].flatten())[0][1])
		           all_rf_dist.append(numpy.mean(numpy.power(r[i].flatten()-r[j].flatten(),2)))				
		
		idx = (numpy.argsort(dst))[1]
		nn_orr_diff.append(circular_distance(params[i][3],params[idx][3]))
		nn_corr.append(numpy.corrcoef(r[i].flatten(),r[idx].flatten())[0][1])
		nn_rf_dist.append(numpy.mean(numpy.power(r[i].flatten()-r[idx].flatten(),2)))

		
    pylab.figure()
    pylab.title('Nearest neighbour orrientation difference')
    pylab.hist([nn_orr_diff,all_orr_diff],normed=True)
    
    pylab.figure()
    pylab.title('Nearest neighbour RFs correlation')
    pylab.hist([nn_corr,all_corr],normed=True)
    
    pylab.figure()
    pylab.title('Nearest neighbour RF distance')
    pylab.hist([nn_rf_dist,all_rf_dist],normed=True)
    
    
	
    #angle_dif50 = []
    #angle50 = []
    #corr50 = []
    #for i in xrange(0,len(new_rfs)):
	#for j in xrange(i+1,len(new_rfs)):
	      #dist = distance(locations,new_rfs_idx[i],new_rfs_idx[j])
	      
	      #if (dist < 50) and (circular_distance(params[i][3],params[j][3])<numpy.pi/6):
		        #a = numpy.arccos((locations[new_rfs_idx[j]][0]-locations[new_rfs_idx[i]][0])/dist) 
			#a = a * numpy.sign(locations[new_rfs_idx[j]][1]-locations[new_rfs_idx[i]][1])
			
			#if a < 0: 
			   #a = a + numpy.pi
			
			#angle_dif50.append(a)
		        #angle50.append(params[i][3])
			#corr50.append(numpy.corrcoef(new_rfs[i].flatten(),new_rfs[j].flatten())[0][1])
    
    
    #angle_dif100 = []
    #angle100 = []
    #corr100= []
    #for i in xrange(0,len(new_rfs)):
	#for j in xrange(i+1,len(new_rfs)):
	      #dist = distance(locations,new_rfs_idx[i],new_rfs_idx[j])
	      
	      #if (dist < 100) and (circular_distance(params[i][3],params[j][3])<numpy.pi/6):
			#a = numpy.arccos((locations[new_rfs_idx[j]][0]-locations[new_rfs_idx[i]][0])/dist) 
			#a = a * numpy.sign(locations[new_rfs_idx[j]][1]-locations[new_rfs_idx[i]][1])
			
			#if a < 0: 
			   #a = a + numpy.pi			
			
			#angle_dif100.append(a)
		        #angle100.append(params[i][3])
    			#corr100.append(numpy.corrcoef(new_rfs[i].flatten(),new_rfs[j].flatten())[0][1])
    #data=[]
    #dataset = loadSimpleDataSet("Mice/2009_11_04/region3_stationary_180_15fr_103cells_on_response_spikes",1800,103)
    #(index,data) = dataset
    #index+=1
    #dataset = (index,data)
    #dataset = averageRangeFrames(dataset,0,1)
    #dataset = averageRepetitions(dataset)
    #dataset = generateTrainingSet(dataset)
    #(a,v) = compute_average_min_max(dataset)
    #dataset = normalize_data_set(dataset,a,v)
    #data.append(dataset) 
    
    #cor_orig = []
    #for i in xrange(0,len(new_rfs)):
	#for i in xrange(0,len(new_rfs)):
		#for j in xrange(i+1,len(new_rfs)):
		#cor_orig.append(numpy.corrcoef(dataset[:,new_rfs_idx[i]].T,dataset[:,new_rfs_idx[j]].T)[0][1])
    
    print len(phase_diff_of_colinear20)
    pylab.figure()
    pylab.title('Histogram of phase difference of co-oriented proximite <0.1 neurons')
    pylab.hist(phase_diff_of_colinear20)
    
    pylab.figure()
    pylab.title('Histogram of phase difference of co-oriented proximite <0.2 neurons')
    pylab.hist(phase_diff_of_colinear30)

    pylab.figure()
    pylab.title('Histogram of phase difference of co-oriented proximite <0.3 neurons')
    pylab.hist(phase_diff_of_colinear50)
    
    pylab.figure()
    pylab.title('Histogram of phase difference of co-oriented proximite <0.4 neurons')
    pylab.hist(phase_diff_of_colinear100)
    
    
    
    pylab.figure()
    pylab.title('Histogram of phase difference of co-oriented proximite <0.1 neurons normalized against histogram of all couples')
    (h,b) = numpy.histogram(phase_diff_of_colinear30)
    (h2,b) = numpy.histogram(phase_diff_of_colinear1000)
    pylab.plot((h*1.0/numpy.sum(h))/(h2*1.0/numpy.sum(h2)))
    
    pylab.figure()
    pylab.title('Histogram of phase difference of co-oriented proximite <0.1 neurons normalized against histogram of all couples')
    (h,b) = numpy.histogram(phase_diff_of_colinear30)
    (h2,b) = numpy.histogram(phase_diff_of_colinear1000)
    pylab.plot((h*1.0/numpy.sum(h))-(h2*1.0/numpy.sum(h2)))
    
    pylab.figure()
    pylab.title('Histogram of phases')
    a = numpy.concatenate([numpy.mat(param[0])[:,5].flatten(),numpy.mat(param[1])[:,5].flatten(),numpy.mat(param[2])[:,5].flatten()],axis=1).flatten()
    pylab.hist(a.T)
    
    import contrib.jacommands
    pylab.figure()
    pylab.title('Correlation between distance and raw RFs average distance')
    pylab.plot(d,rf_dist,'ro')
    pylab.plot(d,contrib.jacommands.weighted_local_average(d,rf_dist,30),'go')
    
    pylab.figure(facecolor='w')
    pylab.title('Correlation between distance and raw RFs correlations')
    ax = pylab.axes() 
    ax.plot(d,c,'ro')
    ax.plot(d,contrib.jacommands.weighted_local_average(d,c,30),'go')
    ax.axhline(0,linewidth=4)
    for tick in ax.xaxis.get_major_ticks():
  	  tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
    	tick.label1.set_fontsize(18)
    pylab.savefig('RFsCorrelationsVsDistance.png')
	
    pylab.xlabel("distance",fontsize=18)
    pylab.ylabel("correlation coefficient",fontsize=18)
    #pylab.plot(d,contrib.jacommands.weighted_local_average(d,numpy.abs(c),30),'bo')
    
    pylab.figure()
    pylab.title('Correlation between distance and centered raw RFs correlations')
    pylab.plot(d,c_cut,'ro')
    pylab.plot(d,contrib.jacommands.weighted_local_average(d,c_cut,30),'go')
    pylab.plot(d,contrib.jacommands.weighted_local_average(d,numpy.abs(c_cut),30),'bo')
    
    pylab.figure()
    pylab.title('Correlation between distance and orr preference')
    pylab.plot(d,orr_diff,'ro')
    pylab.axhline(numpy.pi/4)
    pylab.plot(d,contrib.jacommands.weighted_local_average(d,orr_diff,30),'go')
    
    #pylab.figure()
    #pylab.title('Correlation between firing rate correlations and raw RFs correlations')
    #pylab.plot(c,cor_orig,'ro')
    
    #pylab.figure()
    #pylab.title('Correlation between firing rate correlations and distance')
    #pylab.plot(d,cor_orig,'ro')
    
    #pylab.figure()
    #pylab.title('Angular difference against orientation of proximit (<50) co-oriented pairs of cells')
    #pylab.scatter(angle50,angle_dif50,s=numpy.abs(corr50)*100,marker='o',c='r',cmap=pylab.cm.RdBu)
    #pylab.xlabel("average orienation of pair")
    #pylab.ylabel("angular difference")
    	 
    #pylab.figure()
    #pylab.title('Angular difference against orientation of proximit (<100) co-oriented pairs of cells')
    #pylab.scatter(angle100,angle_dif100,s=numpy.abs(corr100)*100,marker='o',cmap=pylab.cm.RdBu)
    #pylab.xlabel("average orienation of pair")
    #pylab.ylabel("angular difference")
	 
    pylab.figure()
    pylab.title('Histogram of orientations')
    pylab.hist(numpy.matrix(params)[:,3])
	 
def distance(locations,x,y):
    return  numpy.sqrt(numpy.power(locations[x][0] - locations[y][0],2)+numpy.power(locations[x][1] - locations[y][1],2))
  	 
def circular_distance(angle_a,angle_b):
    c= abs(angle_a - angle_b)
    if c > numpy.pi/2:
       c = numpy.pi-c
    return c		 

def RF_corr_centered(RF1,RF2,fraction,display=True):
    sx,sy = numpy.shape(RF1)	
	
    X = numpy.zeros((sx,sy))
    Y = numpy.zeros((sx,sy))
        
    for x in xrange(0,sx):
        for y in xrange(0,sy):
            X[x][y] = x
            Y[x][y] = y
    
    cgs = []
    RFs=[]
    
    cg1x = numpy.round(numpy.sum(numpy.sum(numpy.multiply(X,numpy.power(RF1,2))))/numpy.sum(numpy.sum(numpy.power(RF1,2))))
    cg1y = numpy.round(numpy.sum(numpy.sum(numpy.multiply(Y,numpy.power(RF1,2))))/numpy.sum(numpy.sum(numpy.power(RF1,2))))
    cg2x = numpy.round(numpy.sum(numpy.sum(numpy.multiply(X,numpy.power(RF2,2))))/numpy.sum(numpy.sum(numpy.power(RF2,2))))
    cg2y = numpy.round(numpy.sum(numpy.sum(numpy.multiply(Y,numpy.power(RF2,2))))/numpy.sum(numpy.sum(numpy.power(RF2,2))))
    
    
    RF1c = RF1[cg1x-sx*fraction:cg1x+sx*fraction,cg1y-sy*fraction:cg1y+sy*fraction]
    RF2c = RF2[cg2x-sx*fraction:cg2x+sx*fraction,cg2y-sy*fraction:cg2y+sy*fraction]
    
    if display:
	pylab.figure()
	pylab.subplot(2,1,1)
	pylab.imshow(RF1c,cmap=pylab.cm.RdBu)
	pylab.subplot(2,1,2)
	pylab.imshow(RF2c,cmap=pylab.cm.RdBu)
    
    return numpy.corrcoef(RF1c.flatten(),RF2c.flatten())[0][1]

def centre_of_gravity(matrix):
    sx,sy = numpy.shape(matrix)

    m = matrix*(numpy.abs(matrix)>(0.5*numpy.max(numpy.abs(matrix))))	
    X = numpy.tile(numpy.arange(0,sx,1),(sy,1))	
    Y = numpy.tile(numpy.arange(0,sy,1),(sx,1)).T

    x = numpy.sum(numpy.multiply(X,numpy.power(m,2)))/numpy.sum(numpy.power(m,2))
    y = numpy.sum(numpy.multiply(Y,numpy.power(m,2)))/numpy.sum(numpy.power(m,2))
    
    return (x,y)
	

def low_power(image,t): 
    z = numpy.fft.fft2(image)
    z = numpy.fft.fftshift(z)
    y = numpy.zeros(numpy.shape(z))
    (x,trash) = numpy.shape(y)
    c = x/2
    
    for i in xrange(0,x):
	for j in xrange(0,x):
	    if numpy.sqrt((c-i)*(c-i) + (c-j)*(c-j)) <= t:
	       y[i,j]=1.0
    z = numpy.multiply(z,y)
    z = numpy.fft.ifftshift(z)
    #pylab.figure()
    #pylab.imshow(y)
    #pylab.figure()
    #pylab.imshow(image)
    #pylab.colorbar()
    #pylab.figure()
    #pylab.imshow(numpy.fft.ifft2(z).real)
    #pylab.colorbar()
    return contrast(numpy.fft.ifft2(z).real) 
    
def band_power(image,t,t2): 
    z = numpy.fft.fft2(image)
    z = numpy.fft.fftshift(z)
    y = numpy.zeros(numpy.shape(z))
    (x,trash) = numpy.shape(y)
    c = x/2
    for i in xrange(0,x):
	for j in xrange(0,x):
	    if numpy.sqrt((c-i)*(c-i) + (c-j)*(c-j)) <= t:
	       if numpy.sqrt((c-i)*(c-i) + (c-j)*(c-j)) >= t2:		    
	       		y[i,j]=1.0
    z = numpy.multiply(z,y)
    z = numpy.fft.ifftshift(z)
    return contrast(numpy.fft.ifft2(z).real) 
    

def contrast(image):
    im = image - numpy.mean(image)
    return numpy.sqrt(numpy.mean(numpy.power(im,2)))	


def run_LIP():
	import scipy
	from scipy import linalg
	f = open("results.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[0]
	rfs = node.children[0].data["ReversCorrelationRFs"]
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
	pred_val_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])
	
	training_set = numpy.array(node.children[0].data["LaterTrainingSet"])
	validation_set = numpy.array(node.children[0].data["LaterValidationSet"])
	m = node.children[0].data["LaterModel"]
	#training_set = node.data["training_set"]
	#validation_set = node.data["validation_set"]
        training_inputs = numpy.array(node.data["training_inputs"])
	validation_inputs = numpy.array(node.data["validation_inputs"])
	raw_validation_set = node.data["raw_validation_set"]

	for i in xrange(0,len(raw_validation_set)):
	    raw_validation_set[i] = numpy.array(m.returnPredictedActivities(numpy.mat(raw_validation_set[i])))

		
        #discard low image mean images
	image_mean=[]	
	for i in xrange(0,len(validation_inputs)):
	    image_mean.append(numpy.mean(validation_inputs[i]))
	idx = numpy.argsort(image_mean)[0:14]
	#idx=[]
	
	print idx
	print "Deleting trials with low mean of images"
	validation_inputs = numpy.delete(validation_inputs, idx, axis = 0)	
	validation_set = numpy.delete(validation_set, idx, axis = 0)
	pred_val_act = numpy.delete(pred_val_act, idx, axis = 0)

	
	for i in xrange(0,len(raw_validation_set)):
	    raw_validation_set[i] = numpy.delete(raw_validation_set[i], idx, axis = 0)	
		
	#compute neurons mean before normalization
	neuron_mean = numpy.mean(training_set,axis=0)
	neuron_mean_val = numpy.mean(validation_set,axis=0)
	

	#training_set = training_set - numpy.min(training_set) 
	#validation_set = validation_set - numpy.min(training_set)
	#pred_act_t_a = pred_act_t - numpy.min(pred_act_t)
	#print numpy.sum(((training_set_a >= 0)*1.0))
	#print numpy.sum(((pred_act_t_a >= 0)*1.0))
	
	#training_set_a = numpy.multiply(training_set_a,((training_set_a > 0)*1.0))
	#pred_act_t_a = numpy.multiply(pred_act_t_a,((pred_act_t_a > 0)*1.0))
	#print numpy.sum(((training_set_a >= 0)*1.0))
	#print numpy.sum(((pred_act_t_a >= 0)*1.0))


	#of = run_nonlinearity_detection(numpy.mat(training_set),pred_act,10,False)
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act))
	pred_act_t = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
	pred_val_act_t= apply_sigmoid_output_function(numpy.mat(pred_val_act),ofs)
	
	
	pylab.figure()
	pylab.hist(training_set.flatten())
		
	
	
    	(num_pres,num_neurons) = numpy.shape(training_set)
	raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
	
	signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, pred_act, pred_val_act)
	signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, training_set, validation_set, pred_act_t, pred_val_act_t)
	
	
	print "Mean Reg. pseudoinverse prediction power on training set / validation set: ", numpy.mean(training_prediction_power) , " / " , numpy.mean(validation_prediction_power)
	print "Mean Reg. pseudoinverse  prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(validation_prediction_power_t)
	
	
	
	corr_coef=[]
	for i in xrange(0,len(rfs)):
	    corr_coef.append(numpy.corrcoef(pred_act_t.T[i], training_set.T[i])[0][1])
	
	print "The mean correltation coefficient : ", numpy.mean(corr_coef)
	print "Mean variance of training set:", numpy.mean(numpy.power(numpy.std(training_set,axis=0),2))
	print "Mean variance of validation set:", numpy.mean(numpy.power(numpy.std(validation_set,axis=0),2))
	

	val_corr_coef = []
	measured_neuron_sparsity = [] 
	predicted_neuron_sparsity = []
	
        for i in xrange(0,num_neurons):
	    measured_neuron_sparsity.append(numpy.power(numpy.mean(training_set.T[i]),2) / numpy.mean(numpy.power(training_set.T[i],2)))
	    predicted_neuron_sparsity.append(numpy.power(numpy.mean(pred_act_t.T[i]),2) / numpy.mean(numpy.power(pred_act_t.T[i],2)))
	    val_corr_coef.append(numpy.corrcoef(pred_val_act_t.T[i], validation_set.T[i])[0][1])
	
	measured_pop_sparsity = [] 
	predicted_pop_sparsity = []
	for i in xrange(0,num_pres):
	    measured_pop_sparsity.append(numpy.power(numpy.mean(training_set[i]),2) / numpy.mean(numpy.power(training_set[i],2)))
	    predicted_pop_sparsity.append(numpy.power(numpy.mean(pred_act_t[i]),2) / numpy.mean(numpy.power(pred_act_t[i],2)))	
	
	pylab.figure()
	pylab.title('The sparsity of measured and predicted activity per neurons')
	pylab.hist(numpy.vstack([numpy.array(measured_neuron_sparsity),numpy.array(predicted_neuron_sparsity)]).T,bins=numpy.arange(0,1.01,0.1),label=['measured','predicted'])
	pylab.axvline(numpy.mean(measured_neuron_sparsity),color='b')
	pylab.axvline(numpy.mean(predicted_neuron_sparsity),color='g')
	pylab.legend()
	
	pylab.figure()
	pylab.title('The sparsity of measured and predicted activity per population')
	pylab.hist(numpy.vstack([numpy.array(measured_pop_sparsity),numpy.array(predicted_pop_sparsity)]).T,bins=numpy.arange(0,1.01,0.1),label=['measured','predicted'])
	pylab.axvline(numpy.mean(measured_pop_sparsity),color='b')
	pylab.axvline(numpy.mean(predicted_pop_sparsity),color='g')
	pylab.legend()
	
	pylab.figure()
	
	
	print "The mean correlation coeficient on validation set: ", numpy.mean(val_corr_coef)
	pylab.figure()
	pylab.title("Histogram of neural response means")
	pylab.hist(neuron_mean)
    	
	pylab.figure()
	pylab.title("Histogram of training prediction powers")
	pylab.hist(training_prediction_power_t)
    	
	pylab.figure()
	pylab.title("Histogram of validation prediction powers")
	pylab.hist(validation_prediction_power_t)
    	
	
	#discard low correlation neurons
	r=[]
	corr=[]
	tresh=[]
	print training_prediction_power_t
        for i in xrange(0,30):
		f = numpy.nonzero((numpy.array(training_prediction_power_t) < ((i-10)/30.0))*1.0)[0]
	        tresh.append(((i-10.0)/50.0))
		training_set_good = numpy.delete(training_set, f, axis = 1)
		pred_act_good = numpy.delete(pred_act_t, f, axis = 1)
		validation_set_good = numpy.delete(validation_set, f, axis = 1)
    		pred_val_act_good = numpy.delete(pred_val_act_t, f, axis = 1)
		(rank,correct,tr) = performIdentification(validation_set_good,pred_val_act_good)
		r.append(numpy.mean(rank))
		corr.append(correct)
	
	f = numpy.nonzero((numpy.array(training_prediction_power_t) < 0.1)*1.0)[0]
	f = []
	print f
	training_set = numpy.delete(training_set, f, axis = 1)
	pred_act = numpy.delete(pred_act, f, axis = 1)
	pred_act_t = numpy.delete(pred_act_t, f, axis = 1)
	validation_set = numpy.delete(validation_set, f, axis = 1)
    	pred_val_act_t = numpy.delete(pred_val_act_t, f, axis = 1)
	pred_val_act = numpy.delete(pred_val_act, f, axis = 1)

	(num_pres,num_neurons) = numpy.shape(training_set)

	
    	(ranks,correct,tr) = performIdentification(validation_set,pred_val_act)
    	print "Correct", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act,2))
	(tf_ranks,tf_correct,pred) = performIdentification(validation_set,pred_val_act_t)
	print "Correct", tf_correct , "Mean rank:", numpy.mean(tf_ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act_t,2))
	
	pylab.figure()
	pylab.title("Ranks histogram")
	pylab.xlabel("ranks")
	pylab.hist(tf_ranks,bins=numpy.arange(0,len(tf_ranks),1))
	
	pylab.figure()
	pylab.xlabel("tresh")
	pylab.ylabel("correct")
	pylab.plot(tresh,corr)
	
	pylab.figure()
	pylab.xlabel("tresh")
	pylab.ylabel("rank")
	pylab.plot(tresh,r)

	errors=[]
	bp=[]
	lp5=[]
	lp6=[]
	lp7=[]
	lp8=[]
	lp9=[]
	lp10=[]
	
	image_contrast=[]
	response_mean=[]
	image_mean=[]
	corr_of_pop_resp=[]
	
	sx,sy = numpy.shape(rfs[0])
	
	if False:	
		for i in xrange(0,len(pred_val_act)):
			errors.append(numpy.sum(numpy.power(pred_val_act_t[i] - validation_set[i],2)) / 
					numpy.sum(numpy.power(validation_set[i] - numpy.mean(validation_set[i]),2)))
				
			corr_of_pop_resp.append(numpy.corrcoef(pred_val_act_t[i],validation_set[i],2)[0][1])
			bp.append(band_power(numpy.reshape(validation_inputs[i],(sx,sy)),7,3))
			lp5.append(low_power(numpy.reshape(validation_inputs[i],(sx,sy)),5))
			lp6.append(low_power(numpy.reshape(validation_inputs[i],(sx,sy)),6))
			lp7.append(low_power(numpy.reshape(validation_inputs[i],(sx,sy)),7))
			lp8.append(low_power(numpy.reshape(validation_inputs[i],(sx,sy)),8))
			lp9.append(low_power(numpy.reshape(validation_inputs[i],(sx,sy)),9))
			lp10.append(low_power(numpy.reshape(validation_inputs[i],(sx,sy)),10))
			image_contrast.append(contrast(numpy.reshape(validation_inputs[i],(sx,sy))))
			response_mean.append(numpy.mean(training_set[i]))
			image_mean.append(numpy.mean(numpy.reshape(validation_inputs[i],(sx,sy))))
 	
	

		pylab.figure()
		pylab.title("Correlation between prediction error and contrast of band-passed images")
		pylab.plot(errors,bp,'ro')
		pylab.xlabel("prediction error")
		pylab.ylabel("band pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between prediction error and contrast of low-passed images")
		pylab.plot(errors,lp7,'ro')
		pylab.xlabel("prediction error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between prediction error and basic contrast of images")
		pylab.plot(errors,image_contrast,'ro')
		pylab.xlabel("prediction error")
		pylab.ylabel("basic contrast")
	
		pylab.figure()
		pylab.title("Correlation between correlation of pop resp and contrast of band-passed images")
		pylab.plot(corr_of_pop_resp,bp,'ro')
		pylab.xlabel("correlation of pop resp")
		pylab.ylabel("band pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between correlation of pop resp and contrast of low-passed images")
		pylab.plot(corr_of_pop_resp,lp7,'ro')
		pylab.xlabel("correlation of pop resp")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between correlation of pop resp and basic contrast of images")
		pylab.plot(corr_of_pop_resp,image_contrast,'ro')
		pylab.xlabel("correlation of pop resp")
		pylab.ylabel("basic contrast")
	
		pylab.figure()
		pylab.xlabel("neuronal response mean")
		pylab.ylabel("correlation coef")
		pylab.plot(neuron_mean,corr_coef,'ro')
		
		pylab.figure()
		pylab.hist(tf_ranks,bins=numpy.arange(0,len(tf_ranks),1))
		pylab.title("Histogram of ranks after application of transfer function")
		
		pylab.figure()
		pylab.title("Correlation between correlation of pop resp and rank")
		pylab.plot(corr_of_pop_resp,tf_ranks,'ro')
		pylab.xlabel("correlation of pop resp")
		pylab.ylabel("rank") 
		
		pylab.figure()
		pylab.title("Correlation between rand and  prediction error")
		pylab.plot(tf_ranks,errors,'ro')
		pylab.ylabel("prediction error")
		pylab.xlabel("rank")
	
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of band-passed images")
		pylab.plot(tf_ranks,bp,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("band pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of low-passed images, tresh=5")
		pylab.plot(tf_ranks,lp5,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of low-passed images, tresh=6")
		pylab.plot(tf_ranks,lp6,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of low-passed images, tresh=7")
		pylab.plot(tf_ranks,lp7,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of low-passed images, tresh=8")
		pylab.plot(tf_ranks,lp8,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of low-passed images, tresh=9")
		pylab.plot(tf_ranks,lp9,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and contrast of low-passed images, tresh=10")
		pylab.plot(tf_ranks,lp10,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("low pass contrast")
		
		pylab.figure()
		pylab.title("Correlation between rank error and basic contrast of images")
		pylab.plot(tf_ranks,image_contrast,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("basic contrast")
		
		print "Bad images"
		print image_mean[11]
		
		pylab.figure()
		pylab.title("Correlation between rank error and mean of images")
		pylab.plot(tf_ranks,image_mean,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("image mean")
	
		pylab.figure()
		pylab.title("Correlation between rank error and measured response mean")
		pylab.plot(tf_ranks,response_mean,'ro')
		pylab.xlabel("rank error")
		pylab.ylabel("measured response mean")
		
		pylab.figure()
		pylab.title("Correlation between correlation of pop resp and response mean")
		pylab.plot(corr_of_pop_resp,response_mean,'ro')
		pylab.xlabel("correlation of pop resp")
		pylab.ylabel("response mean")
		
		pylab.figure()
		pylab.title("Correlation between correlation of pop resp and image mean")
		pylab.plot(corr_of_pop_resp,image_mean,'ro')
		pylab.xlabel("correlation of pop resp")
		pylab.ylabel("image mean")
		
		pylab.figure()
		pylab.title("Correlation between measured response mean and image mean")
		pylab.plot(response_mean,image_mean,'ro')
		pylab.xlabel("response mean")
		pylab.ylabel("image mean")
		
		pylab.figure()
		pylab.title("Correlation between measured response mean and image basic contrast")
		pylab.plot(response_mean,image_contrast,'ro')
		pylab.xlabel("response mean")
		pylab.ylabel("image basic contrast")
		
		pylab.figure()
		pylab.title("Correlation between measured response mean and contrast of low passed image t=7")
		pylab.plot(response_mean,lp7,'ro')
		pylab.xlabel("response mean")
		pylab.ylabel("low passed image contrast")
		
		
		fig = pylab.figure()
		from mpl_toolkits.mplot3d import Axes3D
		ax = Axes3D(fig)
	
		ax.scatter(lp7,image_mean,tf_ranks)
		ax.set_xlabel("contrast")
		ax.set_ylabel("image mean")
		ax.set_zlabel("rank")
		
		fig = pylab.figure()
		ax = Axes3D(fig)
	
		ax.scatter(lp7,corr_of_pop_resp,tf_ranks)
		ax.set_xlabel("contrast")
		ax.set_ylabel("corr_of_pop_resp")
		ax.set_zlabel("rank")
		
	if False:
		(later_pred_act,later_pred_val_act) = later_interaction_prediction(training_set,pred_act_t,validation_set,pred_val_act_t,contrib.dd.DB(None))
		
		#of = run_nonlinearity_detection(numpy.mat(training_set),later_pred_act,10,False)
		training_set += 2.0
		validation_set += 2.0
		ofs = fit_exponential_to_of(numpy.mat(training_set),numpy.mat(later_pred_act)+2.0)
		later_pred_act_t = apply_exponential_output_function(later_pred_act+2.0,ofs)
		later_pred_val_act_t= apply_exponential_output_function(later_pred_val_act+2.0,ofs)
		#later_pred_act_t = later_pred_act
		#later_pred_val_act_t = later_pred_val_act
		
		(ranks,correct,pred) = performIdentification(validation_set,later_pred_val_act+2.0)
		print "After lateral identification> Correct:", correct , "Mean rank:", numpy.mean(ranks)
	
		(tf_ranks,tf_correct,pred) = performIdentification(validation_set,later_pred_val_act_t)
		print "After lateral identification> TFCorrect:", tf_correct , "Mean tf_rank:", numpy.mean(tf_ranks)
	
		#signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, later_pred_act, later_pred_val_act)
		signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, training_set, validation_set, later_pred_act_t, later_pred_val_act_t)
	
	#return
	
	
	#for ii in xrange(0,5):
	#    pylab.figure()
	#    pylab.hist(later_pred_act_t[:,ii].flatten())
	#    pylab.figure()
	#    pylab.hist(training_set[:,ii].flatten())
	 
	g = 1
	for (x,i) in pred:
	    #if x==i: continue
	    pylab.figure()
	    pylab.subplot(3,1,1)
	    pylab.imshow(numpy.reshape(validation_inputs[i],(sx,sy)),vmin=-128,vmax=128,interpolation='nearest',cmap=pylab.cm.gray)
	    pylab.title('Correct')
	    pylab.axis('off')
	    pylab.subplot(3,1,2)
	    pylab.imshow(numpy.reshape(validation_inputs[x],(sx,sy)),vmin=-128,vmax=128,interpolation='nearest',cmap=pylab.cm.gray)
	    pylab.title('Picked')
	    pylab.axis('off')
	    
	    pylab.subplot(3,1,3)
	    pylab.plot(numpy.array(pred_val_act_t)[i],'ro',label='Predicted activity')
	    pylab.plot(validation_set[i],'bo',label='Measured activity')
	    pylab.axhline(y=numpy.mean(validation_set[i]),linewidth=1, color='b')
	    pylab.axhline(y=numpy.mean(pred_val_act_t[i]),linewidth=1, color='r')
	    if x != i:
	       pylab.plot(numpy.array(pred_val_act_t)[x],'go',label='Most similar')
	       pylab.axhline(y=numpy.mean(numpy.array(pred_val_act_t)[x]),linewidth=1, color='g')
	    pylab.legend()
	    
	    for j in xrange(0,len(validation_set[0])):
		if(abs(numpy.array(pred_val_act_t)[i][j] - validation_set[i][j]) < abs(numpy.array(pred_val_act_t)[x][j] - validation_set[i][j])):
		   	pylab.axvline(j)	
	    
            g+=1


def performIdentification(responses,model_responses):
    correct=0
    ranks=[]
    pred=[]
    for i in xrange(0,len(responses)):
        tmp = []
        for j in xrange(0,len(responses)):
            tmp.append(numpy.sqrt(numpy.mean(numpy.power(numpy.mat(responses)[i]-model_responses[j],2))))
        x = numpy.argmin(tmp)
	z = tmp[i]
	ranks.append(numpy.nonzero((numpy.sort(tmp)==z)*1.0)[0][0])
        if (x == i): correct+=1
	pred.append((x,i))
    return (ranks,correct,pred)


	
	
	
	

#from pygene.organism import Organism
#from pygene.gene import FloatGene
#class ComplexCellOrganism(Organism):
	#training_set = []
	#training_inputs = []
	
	#def fitness(self):
		#z,t = numpy.shape(self.training_inputs) 
		#x =  self[str(0)]
		#y =  self[str(1)]
		#sigma = self[str(2)]*0.1
		#angle = self[str(3)]*numpy.pi
		#p =  self[str(4)]*numpy.pi*2
		#f = self[str(5)]*10
		#ar = self[str(6)]*2.5
		#alpha = self[str(7)]
		#dx = numpy.sqrt(t)
		#dy = dx
		#g =  numpy.mat(Gabor(bounds=BoundingBox(radius=0.5),frequency=f,x=x-0.5,y=y-0.5,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,aspect_ratio=ar)() * alpha)
		#r1 = self.training_inputs * g.flatten().T
		#return numpy.mean(numpy.power(r1-self.training_set,2))  

#rand =numbergen.UniformRandom(seed=513)
#class CCGene(FloatGene):
      #randMin=0.0
      #randMax=1.0
      ##def mutate(self):
##	  self.value = self.value + self.value*2.0*(0.5-rand())

#def GeneticAlgorithms():
    #from pygene.gamete import Gamete
    #from pygene.population import Population
    
    #f = open("modelfitDB2.dat",'rb')
    #import pickle
    #dd = pickle.load(f)
    #training_set = dd.children[0].data["training_set"][0:1800,:]
    #training_inputs = dd.children[0].data["training_inputs"][0:1800,:]
    
    ##dd = contrib.dd.DB(None)
    
    ##(sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node) = sortOutLoading(dd)
    
    
    
    #genome = {}
    #for i in range(8):
    	#genome[str(i)] = CCGene
    
    #ComplexCellOrganism.genome = genome
    #ComplexCellOrganism.training_set = numpy.mat(training_set)[:,0]
    #ComplexCellOrganism.training_inputs = numpy.mat(training_inputs)
    	  
    #class CPopulation(Population):
	  #species = ComplexCellOrganism
	  #initPopulation = 200
	  #childCull = 100
	  #childCount = 500
	  #incest=10
	  #i = 0
	  
    #pop = CPopulation()
    
    #pylab.ion()
    #pylab.hold(False)
    #pylab.figure()
    #pylab.show._needmain=False
    #pylab.show()
    #pylab.figure()
    #while True:
    	#pop.gen()
	#best = pop.best()
	#print "fitness:" , (best.fitness())
	#z,t = numpy.shape(training_inputs) 
	#x =  best[str(0)]
	#y =  best[str(1)]
	#sigma = best[str(2)]*0.1
	#angle = best[str(3)]*numpy.pi
	#p =  best[str(4)]*numpy.pi*2
	#f = best[str(5)]*10
	#ar = best[str(6)]*2.5
	#alpha = best[str(7)]
	#dx = numpy.sqrt(t)
	#dy = dx
	#g =  numpy.mat(Gabor(bounds=BoundingBox(radius=0.5),frequency=f,x=x-0.5,y=y-0.5,xdensity=dx,ydensity=dy,size=sigma,orientation=angle,phase=p,aspect_ratio=ar)() * alpha)
	#m=numpy.max([numpy.abs(numpy.min(g)),numpy.abs(numpy.max(g))])
	#pylab.subplot(2,1,1)
	#pylab.imshow(g,vmin=-m,vmax=m,cmap=pylab.cm.RdBu,interpolation='nearest')
	
	#pylab.show._needmain=False
    	#pylab.show()

def runSurrondStructureDetection():
    f = open("results.dat",'rb')
    import pickle
    dd = pickle.load(f)
    node = dd.children[0]
    act = node.data["training_set"]
    val_act = node.data["validation_set"]
    node = node.children[0]
    pred_act  = numpy.array(node.data["ReversCorrelationPredictedActivities"])
    pred_val_act  = numpy.array(node.data["ReversCorrelationPredictedValidationActivities"])
		
			
    dataset = loadSimpleDataSet("Mice/2009_11_04/region3_stationary_180_15fr_103cells_on_response_spikes",1800,103)
    (index,data) = dataset
    index+=1
    dataset = (index,data)
    valdataset = loadSimpleDataSet("Mice/2009_11_04/region3_50stim_10reps_15fr_103cells_on_response_spikes",50,103,10)
    #(valdataset,trash) = splitDataset(valdataset,40)
    
    
    training_inputs=generateInputs(dataset,"/home/antolikjan/topographica/topographica/Flogl/DataOct2009","/20090925_image_list_used/image_%04d.tif",__main__.__dict__.get('density', 20),1.8,offset=1000)
    
    validation_inputs=generateInputs(valdataset,"/home/antolikjan/topographica/topographica/Mice/2009_11_04/","/20091104_50stimsequence/50stim%04d.tif",__main__.__dict__.get('density', 20),1.8,offset=0)
    #validation_inputs = validation_inputs[0:40]
    
    (sizex,sizey) = numpy.shape(training_inputs[0])
    #mask = numpy.zeros(numpy.shape(training_inputs[0]))
    #mask[sizex*0.1:sizex*0.9,sizey*0.6:sizey*0.9]=1.0
    
    #for i in xrange(0,1800):
    #	training_inputs[i] = training_inputs[i][:,sizey/2:sizey] 
    #for i in xrange(0,50):	
    # 	validation_inputs[i] = validation_inputs[i][:,sizey/2:sizey]
    
    (sizex,sizey) = numpy.shape(training_inputs[0])
    
    
    ofs = fit_sigmoids_to_of(numpy.mat(act),numpy.mat(pred_act))
    pred_act = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
    pred_val_act= apply_sigmoid_output_function(numpy.mat(pred_val_act),ofs)

    
    print sizex,sizey
    cc = 0.7
    #print pred_act
    new_target_act =  numpy.divide(act+cc,pred_act+cc)
    new_val_target_act =  numpy.divide(val_act+cc,pred_val_act+cc)
    
    training_inputs = generate_raw_training_set(training_inputs)
    validation_inputs = generate_raw_training_set(validation_inputs)
        
    print "Mins"
    print numpy.min(pred_act)
    print numpy.min(pred_val_act)
    print numpy.min(act)
    print numpy.min(val_act)
    
    
    (e,te,c,tc,RFs,pa,pva,corr_coef,corr_coef_tf) = regulerized_inverse_rf(training_inputs,new_target_act,sizex,sizey,__main__.__dict__.get('Alpha',50),numpy.mat(validation_inputs),numpy.mat(new_val_target_act),contrib.dd.DB(None),True)
    
    ofs = run_nonlinearity_detection(numpy.mat(act+cc),numpy.mat(numpy.multiply(pred_act+cc,pa)))
    pa_t = apply_output_function(numpy.multiply(pred_act+cc,pa),ofs)
    pva_t = apply_output_function(numpy.multiply(pred_val_act+cc,pva),ofs)
    
    print numpy.min(pva)
    print numpy.min(pva_t)
    
    (ranks,correct,pred) = performIdentification(val_act+cc,pred_val_act+cc)
    print "Without surround", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(val_act+cc - pred_val_act-cc,2))
    
    (ranks,correct,pred) = performIdentification(val_act+cc,numpy.multiply(pred_val_act+cc,pva))
    print "With surround", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(val_act+cc - numpy.multiply(pred_val_act+cc,pva),2))
    
    (ranks,correct,pred) = performIdentification(val_act+cc,numpy.multiply(pred_val_act+cc,pva_t))
    print "With surround+ TF", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(val_act+cc - numpy.multiply(pred_val_act+cc,pva_t),2))
    
    
    params={}
    params["SurrAnalysis"] = True
    node = node.get_child(params)
    node.add_data("TrainingInputs",training_inputs,force=True)
    node.add_data("ValidationInputs",validation_inputs,force=True)
    params={}
    params["Alpha"] = __main__.__dict__.get('Alpha',50)
    params["Density"] = __main__.__dict__.get('density', 20)
    node = node.get_child(params)
    node.add_data("SurrRFs",RFs,force=True)	
    f = open("results.dat",'wb')
    pickle.dump(dd,f,-2)
    f.close()


def runSTCandSTAtest():
    f = open("modelfitDB2.dat",'rb')
    import pickle
    dd = pickle.load(f)
    
    
    
    STCact = dd.children[6].data["STCact"]
    STCrfs = dd.children[6].data["STCrfs"]
    predicted_activities = dd.children[0].children[0].data["ReversCorrelationPredictedActivities"]
    tf_predicted_activities = dd.children[0].children[0].data["ReversCorrelationPredictedActivities+TF"]
    predicted_validation_activities = dd.children[0].children[0].data["ReversCorrelationPredictedValidationActivities"]
    tf_validation_predicted_activities = dd.children[0].children[0].data["ReversCorrelationPredictedValidationActivities+TF"]
    target_act = dd.children[6].data["training_set"]
    target_val_act = dd.children[6].data["validation_set"]
    training_inputs = dd.children[6].data["training_inputs"]
    validation_inputs = dd.children[6].data["validation_inputs"]
    
    model_predicted_activities = numpy.mat(numpy.zeros(numpy.shape(predicted_activities)))
    model_validation_predicted_activities = numpy.mat(numpy.zeros(numpy.shape(predicted_validation_activities)))
    
    
    for (rfs,i) in zip(STCrfs,xrange(0,len(STCrfs))):
	(ei,vv,avv,em,ep) = rfs	    
	a = predicted_activities[:,i]
	a_v = predicted_validation_activities[:,i]
	for j in avv:
	    r = ei[j,:].real
	    o = run_nonlinearity_detection((training_inputs*r.T),numpy.mat(target_act)[:,i],10,display=False)
	    act = apply_output_function(training_inputs*r.T,o)
	    val_act = apply_output_function(validation_inputs*r.T,o)
	    a = numpy.concatenate((a,act),axis=1)
	    a_v = numpy.concatenate((a_v,val_act),axis=1)
	mf = ModelFit()
	mf.learning_rate = __main__.__dict__.get('lr',0.01)
	mf.epochs=__main__.__dict__.get('epochs',100)
	mf.num_of_units = 1
	mf.init()
	
	
	(err,stop,min_errors) = mf.trainModel(a,numpy.mat(target_act)[:,i],a_v,numpy.mat(target_val_act)[:,i])
	
	print numpy.shape(numpy.mat(model_predicted_activities)[:,i])
	print numpy.shape(mf.returnPredictedActivities(mat(a))[:,0]) 
	model_predicted_activities[:,i] = mf.returnPredictedActivities(mat(a))[:,0]
	model_validation_predicted_activities[:,i] = mf.returnPredictedActivities(mat(a_v))[:,0]

    #(ranks,correct,cc) = performIdentification(target_act,model_predicted_activities)
    #print "After lateral identification> TFCorrect:", tf_correct , "Mean tf_rank:", numpy.mean(tf_ranks)
    (ranks,correct,cc) = performIdentification(target_val_act,predicted_validation_activities)
    print "Simple Correct:", correct , "Mean tf_rank:", numpy.mean(ranks), "Percentage:" ,correct/(len(ranks)*1.0)*100 ,"%"
    (ranks,correct,cc) = performIdentification(target_val_act,model_validation_predicted_activities)
    print "Simple + Complex Correct:", correct , "Mean tf_rank:", numpy.mean(ranks), "Percentage:" ,correct/(len(ranks)*1.0)*100 ,"%"	
    
    ofs = run_nonlinearity_detection(numpy.mat(target_act),model_predicted_activities,10,display=True)
    pred_act_t = apply_output_function(model_predicted_activities,ofs)
    pred_val_act_t= apply_output_function(model_validation_predicted_activities,ofs)

    (ranks,correct,cc) = performIdentification(target_val_act,pred_val_act_t)
    print "Simple + Complex + TF Correct:", correct , "Mean tf_rank:", numpy.mean(ranks), "Percentage:" ,correct/(len(ranks)*1.0)*100 ,"%"	
	
	

def analyseInhFiring():
    dataset = loadSimpleDataSet("Mice/2009716_17_03_10/(20090716_17_03_10)-_orientation_classic_region9_15hz_8oris_4grey_2mov_DFOF",6138,27,transpose=True)
    #dataset = loadSimpleDataSet("Mice/2009_11_04/region3_stationary_180_15fr_103cells_on_response_spikes",1800,103,transpose=False)
    ts = generateTrainingSet(dataset)
    
    (x,y) = numpy.shape(ts)
    
    #ts = ts[0:6000,:]
    
    inh = numpy.array([2, 20, 22,23,26])
    inh = inh - 1.0

    exc  = numpy.delete(ts,inh,axis=1)
    inh  = numpy.delete(ts,numpy.delete(numpy.arange(0,y,1),inh),axis=1)	
    
    exc_base = numpy.concatenate((exc[0:93,:] , exc[0:93,-93:]),axis=0)
    inh_base = numpy.concatenate((inh[0:93,:] , inh[0:93,-93:]),axis=0)

    exc = exc[93:-93,:]
    inh = inh[93:-93,:]
    
    print numpy.shape(exc)
    
    exc_average_trace = [numpy.mean(e.reshape(93,64),axis=0) for e in exc.T]
    inh_average_trace = [numpy.mean(i.reshape(93,64),axis=0) for i in inh.T]

    print numpy.shape(exc_average_trace)
    
    pylab.figure()
    pylab.title("Inhibitory neurons")
    for e in exc.T:
	pylab.plot(e)
    pylab.ylim((-0.07,0.2))
    
    pylab.figure()
    pylab.title("Excitatory neurons")
    for i in inh.T:
	pylab.plot(i)
    pylab.ylim((-0.07,0.2))
    
    pylab.figure()
    
    pylab.title("Trace excitatory")
    for e in exc_average_trace:
	pylab.plot(e)
    pylab.ylim((-0.0,0.05))
    
    pylab.figure()
    pylab.title("Trace inhibitory")
    for i in inh_average_trace:
	pylab.plot(i)
    pylab.ylim((-0.0,0.05))
    
    pylab.figure()
    pylab.title("baseline: Excitatory neurons")
    for e in exc_base.T:
	pylab.plot(e)
    pylab.ylim((-0.05,0.2))
  
    pylab.figure()
    pylab.title("baseline: Inhibitory neurons")
    for i in inh_base.T:
	pylab.plot(i)
    pylab.ylim((-0.05,0.2))
    
    
    pylab.figure()
    pylab.title('mean vs max of neurons')
    pylab.plot(numpy.mean(exc.T,axis=1),numpy.max(exc.T,axis=1),'ro')	
    pylab.plot(numpy.mean(inh.T,axis=1),numpy.max(inh.T,axis=1),'go')

    pylab.figure()
    pylab.title('mean vs variance of neurons')
    pylab.plot(numpy.mean(exc.T,axis=1),numpy.var(exc.T,axis=1),'ro')	
    pylab.plot(numpy.mean(inh.T,axis=1),numpy.var(inh.T,axis=1),'go')

    pylab.figure()
    pylab.title('mean triggered vs mean at base')
    pylab.plot(numpy.mean(exc.T,axis=1),numpy.mean(exc_base.T,axis=1),'ro')	
    pylab.plot(numpy.mean(inh.T,axis=1),numpy.mean(inh_base.T,axis=1),'go')


    exc_fft  = [numpy.fft.fft(e) for e in exc.T]
    inh_fft  = [numpy.fft.fft(i) for i in inh.T]

    exc_fft_power  = [numpy.abs(e) for e in exc_fft]
    inh_fft_power  = [numpy.abs(e) for e in inh_fft]

    exc_fft_phase  = [numpy.angle(e) for e in exc_fft]
    inh_fft_phase  = [numpy.angle(e) for e in inh_fft]


    
    
    exc_fft_base  = [numpy.fft.fft(e) for e in exc_base.T]
    inh_fft_base  = [numpy.fft.fft(i) for i in inh_base.T]

    exc_fft_power_base  = [numpy.abs(e) for e in exc_fft_base]
    inh_fft_power_base  = [numpy.abs(e) for e in inh_fft_base]

    exc_fft_phase_base  = [numpy.angle(e) for e in exc_fft_base]
    inh_fft_phase_base  = [numpy.angle(e) for e in inh_fft_base]




    pylab.figure()
    pylab.plot(numpy.mean(exc_fft_power,axis=0))	

    pylab.figure()
    pylab.plot(numpy.mean(inh_fft_power,axis=0))	

    pylab.figure()
    pylab.plot(numpy.mean(exc_fft_phase,axis=0))	

    pylab.figure()
    pylab.plot(numpy.mean(inh_fft_phase,axis=0))
    	
    pylab.figure()
    pylab.title('Power spectrum of baseline of excitatory neurons')
    pylab.plot(numpy.mean(exc_fft_power_base,axis=0))	

    pylab.figure()
    pylab.title('Power spectrum of baseline of inhibitory neurons')
    pylab.plot(numpy.mean(inh_fft_power_base,axis=0))	
    
    
    
    pylab.figure()
    pylab.title('high feq power of baseline vs mean of triggered')
    pylab.plot(numpy.mean(numpy.mat(exc_fft_power_base)[:,7:25],axis=1).T,numpy.mat(exc_fft_power).T[0],'ro')
    pylab.plot(numpy.mean(numpy.mat(inh_fft_power_base)[:,7:25],axis=1).T,numpy.mat(inh_fft_power).T[0],'go')
    
    
    pylab.figure()
    pylab.title('high feq power of triggered vs mean of triggered')
    pylab.plot(numpy.mean(numpy.mat(exc_fft_power)[:,7:25],axis=1).T,numpy.mat(exc_fft_power).T[0],'ro')
    pylab.plot(numpy.mean(numpy.mat(inh_fft_power)[:,7:25],axis=1).T,numpy.mat(inh_fft_power).T[0],'go')
    
    pylab.figure()
    pylab.title('high feq power of triggered vs mean of triggered')
    pylab.plot(numpy.mean(numpy.mat(exc_fft_power)[:,7:50],axis=1).T,numpy.mat(exc_fft_power).T[0],'ro')
    pylab.plot(numpy.mean(numpy.mat(inh_fft_power)[:,7:50],axis=1).T,numpy.mat(inh_fft_power).T[0],'go')

    pylab.figure()
    pylab.title('high feq power of triggered vs mean of triggered')
    pylab.plot(numpy.mean(numpy.mat(exc_fft_power)[:,7:70],axis=1).T,numpy.mat(exc_fft_power).T[0],'ro')
    pylab.plot(numpy.mean(numpy.mat(inh_fft_power)[:,7:70],axis=1).T,numpy.mat(inh_fft_power).T[0],'go')
    
    
    
    pylab.figure()
    pylab.title('fanofactor vs variance of trace')
    pylab.plot(numpy.var(exc_average_trace,axis=1)/numpy.mean(exc_average_trace,axis=1),numpy.var(exc_average_trace,axis=1),'ro')
    pylab.plot(numpy.var(inh_average_trace,axis=1)/numpy.mean(inh_average_trace,axis=1),numpy.var(inh_average_trace,axis=1),'go')
    
    pylab.figure()
    pylab.title('fanofactor vs mean of trace')
    pylab.plot(numpy.var(exc_average_trace,axis=1)/numpy.mean(exc_average_trace,axis=1),numpy.mean(exc_average_trace,axis=1),'ro')
    pylab.plot(numpy.var(inh_average_trace,axis=1)/numpy.mean(inh_average_trace,axis=1),numpy.mean(inh_average_trace,axis=1),'go')
    
    pylab.figure()
    pylab.title('variance vs mean of trace')
    pylab.plot(numpy.var(exc_average_trace,axis=1),numpy.mean(exc_average_trace,axis=1),'ro')
    pylab.plot(numpy.var(inh_average_trace,axis=1),numpy.mean(inh_average_trace,axis=1),'go')
    
    
    pylab.figure()
    pylab.title('fanofactor vs variance of base')
    pylab.plot(numpy.var(exc_base,axis=0)/numpy.mean(exc_base,axis=0),numpy.var(exc_base,axis=0),'ro')
    pylab.plot(numpy.var(inh_base,axis=0)/numpy.mean(inh_base,axis=0),numpy.var(inh_base,axis=0),'go')
    
    pylab.figure()
    pylab.title('fanofactor vs mean of base')
    pylab.plot(numpy.var(exc_base,axis=0)/numpy.mean(exc_base,axis=0),numpy.mean(exc_base,axis=0),'ro')
    pylab.plot(numpy.var(inh_base,axis=0)/numpy.mean(inh_base,axis=0),numpy.mean(inh_base,axis=0),'go')

    pylab.figure()
    pylab.title('variance vs mean of base')
    pylab.plot(numpy.var(exc_base,axis=0),numpy.mean(exc_base,axis=0),'ro')
    pylab.plot(numpy.var(inh_base,axis=0),numpy.mean(inh_base,axis=0),'go')
    
    
    
    
    
    pylab.figure()
    pylab.title('mean vs 1st harmonic of neurons')
    pylab.plot(numpy.mat(exc_fft_power).T[0],numpy.mat(exc_fft_power).T[64],'ro')
    pylab.plot(numpy.mat(inh_fft_power).T[0],numpy.mat(inh_fft_power).T[64],'go')
    
    pylab.figure()
    pylab.title('1st vs 2nd harmonic of neurons')
    pylab.plot(numpy.mat(exc_fft_power).T[64],numpy.mat(exc_fft_power).T[128],'ro')
    pylab.plot(numpy.mat(inh_fft_power).T[64],numpy.mat(inh_fft_power).T[128],'go')

    
    pylab.figure()
    pylab.title('mean/1st harmonic vs 1st/2nd harmonic of neurons')
    pylab.plot(numpy.mat(exc_fft_power).T[0] / numpy.mat(exc_fft_power).T[64], numpy.mat(exc_fft_power).T[64] / numpy.mat(exc_fft_power).T[128] ,'ro')
    pylab.plot(numpy.mat(inh_fft_power).T[0] / numpy.mat(inh_fft_power).T[64], numpy.mat(inh_fft_power).T[64] / numpy.mat(inh_fft_power).T[128] ,'go')
    
    
    
    pylab.figure()
    pylab.title('mean vs power at 1st harmonic of neurons')
    pylab.plot(numpy.mean(exc.T,axis=1),numpy.array(numpy.mat(exc_fft_power).T[64])[0],'ro')
    pylab.plot(numpy.mean(inh.T,axis=1),numpy.array(numpy.mat(inh_fft_power).T[64])[0],'go')
    
    pylab.figure()
    pylab.title('power at harmonic vs phase at harmonic of neurons')
    pylab.plot(numpy.mat(exc_fft_power).T[64],numpy.mat(exc_fft_phase).T[64],'ro')
    pylab.plot(numpy.mat(inh_fft_power).T[64],numpy.mat(inh_fft_phase).T[64],'go')
    
    print zip(numpy.mean(exc_fft_power,axis=0),numpy.arange(0,x,1))[0:200]
    
    return(numpy.mean(inh_fft_power,axis=0))


def activationPatterns():
    from scipy import linalg
    f = open("modelfitDB2.dat",'rb')
    import pickle
    dd = pickle.load(f)
    node = dd.children[0]
    activities = node.data["training_set"]
    validation_activities = node.data["validation_set"]

    
    num_act,len_act = numpy.shape(activities)
    
    CC = numpy.zeros((len_act,len_act))
   
    for a in activities:
	CC = CC + numpy.mat(a).T * numpy.mat(a)
	CC = CC / num_act
	
    v,la = linalg.eigh(CC)
    pylab.figure()	
    pylab.plot(numpy.sort(numpy.abs(v.real[-30:-1])),'ro')
    
    ind = numpy.argsort(numpy.abs(v.real))
    
    pylab.figure()
    pylab.plot(la[ind[-1],:],'ro')
    pylab.figure()
    pylab.plot(la[ind[-2],:],'ro')
    pylab.figure()
    pylab.plot(la[ind[-3],:],'ro')
    
    pylab.figure()
    pylab.plot(numpy.mat(activities)*numpy.mat(la[ind[-1],:]).T) 
    
    pylab.figure()
    pylab.hist(numpy.mat(activities)*numpy.mat(la[ind[-1],:]).T)
    
    pylab.figure()
    pylab.plot(numpy.mat(activities)*numpy.mat(la[ind[-1],:]).T,numpy.mat(activities)*numpy.mat(la[ind[-2],:]).T,'ro')
    
    node.add_data("ActivityPattern",la[ind[-1],:],force=True)
    
    f = open("modelfitDB2.dat",'wb')
    pickle.dump(dd,f,-2)
    f.close()
    
    pred_act=node.children[0].data["ReversCorrelationPredictedActivities"]
    pred_val_act=node.children[0].data["ReversCorrelationPredictedValidationActivities"]	
	
    ofs = fit_sigmoids_to_of(numpy.mat(activities),numpy.mat(pred_act))
    pred_act = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
    pred_val_act= apply_sigmoid_output_function(numpy.mat(pred_val_act),ofs)
	
	
    pylab.figure()	
    print numpy.shape(1-numpy.divide(numpy.sum(numpy.power(activities-pred_act,2),axis=1),numpy.var(activities,axis=1)))
    print numpy.shape(numpy.sum(numpy.power(activities-pred_act,2),axis=1))
    print numpy.shape(numpy.var(activities,axis=1)*len_act)
    
    pylab.plot(1-numpy.divide(numpy.sum(numpy.power(activities-pred_act,2),axis=1),numpy.mat(numpy.var(activities,axis=1)*len_act).T).T,numpy.mat(activities)*numpy.mat(la[ind[-1],:]).T,'ro')
    
    (ranks,correct,pred) =  performIdentification(validation_activities,pred_val_act)
    print correct
    
    pylab.figure()
    pylab.plot(ranks,numpy.mat(validation_activities)*numpy.mat(la[ind[-1],:]).T,'ro')
    
    
	
def AdaptationAnalysis():
	import scipy
	from scipy import linalg
	f = open("modelfitDatabase.dat",'rb')
	import pickle
	dd = pickle.load(f)

	rfs_area1  = dd.children[0].children[0].data["ReversCorrelationRFs"]
	rfs_area2  = dd.children[1].children[0].data["ReversCorrelationRFs"]
	pred_act_area1  = dd.children[0].children[0].data["ReversCorrelationPredictedActivities"][0:1260,:]
	pred_act_area2  = dd.children[1].children[0].data["ReversCorrelationPredictedActivities"]
	
	training_set_area1 = dd.children[0].data["training_set"][0:1260,:]
	training_set_area2 = dd.children[1].data["training_set"]

	rfs = numpy.concatenate((rfs_area1,rfs_area2),axis=0)
	pred_act = numpy.mat(numpy.concatenate((pred_act_area1,pred_act_area2),axis=1))
	training_set = numpy.mat(numpy.concatenate((training_set_area1,training_set_area2),axis=1))
	
	#weights = [0.0,1.0,0.5,0.3,0.1,0.05]
	weights = numpy.exp(-numpy.arange(0,100,1.0)/500.0)
	weights = numpy.insert(weights,0,0)
	print weights
	kl = len(weights)-1
	
	hist=[]
	for i in xrange(0,158):
 	    hist.append(scipy.convolve(numpy.array(training_set[:,i])[:,0],weights,mode='valid'))
	
	print numpy.shape(numpy.mat(training_set[kl:,:]))
	print numpy.shape(numpy.mat(numpy.array(hist)).T)
	ofs = run_nonlinearity_detection(numpy.mat(training_set[kl:,:]),numpy.mat(numpy.array(hist)).T,display=True,num_bins=8)
	
	pylab.figure()
	pylab.hist(numpy.array(hist).flatten())
	pylab.figure()
	pylab.hist(numpy.array(training_set).flatten())
	
	pylab.figure()
	for i in xrange(0,103):
	    pylab.subplot(13,13,i+1)
	    errors = numpy.array((training_set[kl:,i] - pred_act[kl:,i]))[:,0]
	    pylab.plot(hist[i],errors,'ro')
	
	pylab.figure()
	for i in xrange(0,103):
	    pylab.subplot(13,13,i+1)
	    t = numpy.array(training_set[kl:,i])[:,0]
	    pylab.plot(hist[i],t,'ro')


	pylab.figure()
	for i in xrange(0,103):
	    pylab.subplot(13,13,i+1)
	    t = numpy.array(pred_act[kl:,i])[:,0]
	    pylab.plot(hist[i],t,'ro')
	
	fig = pylab.figure()
	from mpl_toolkits.mplot3d import Axes3D
	ax = Axes3D(fig)
	ax.scatter(pred_act[kl:,89],hist[89],training_set[kl:,i])
	ax.set_xlabel("predicted activity")
	ax.set_ylabel("history")
	ax.set_zlabel("training_set")
	
	fig = pylab.figure()
	from mpl_toolkits.mplot3d import Axes3D
	ax = Axes3D(fig)
	ax.scatter(pred_act[kl:,88],hist[88],training_set[kl:,i])
	ax.set_xlabel("predicted activity")
	ax.set_ylabel("history")
	ax.set_zlabel("training_set")

	
	#lets do the gradient 
	from scipy.optimize import leastsq
	xs = []
	err = []
	for i in xrange(0,158):
	    rand =numbergen.UniformRandom(seed=513)
	    x0 = [0.7,-1.0,1.6,-1.0]
       	    xopt = leastsq(history_error, x0[:], args=(numpy.array(hist)[i],numpy.array(pred_act)[kl:,i],numpy.array(training_set)[kl:,i]),ftol=0.0000000000000000001,xtol=0.0000000000000001,warning=False)
	    xs.append(xopt[0])
            new_error  = numpy.sum(history_error(xopt[0],numpy.array(hist)[i],numpy.array(pred_act)[kl:,i],numpy.array(training_set)[kl:,i])**2)
	    old_error =  numpy.sum((numpy.array(pred_act)[kl:,i] - numpy.array(training_set)[kl:,i])**2)
	    err.append( (old_error - new_error)/old_error * 100)
	    	
	print "Error decreased by:", numpy.mean(err) , '%'
	
	new_act = []
	for i in xrange(0,158):
	    new_act.append(history_estim(xs[i],numpy.array(hist)[i],numpy.array(pred_act)[kl:,i]))
	    	
	new_act = numpy.mat(new_act).T
	print numpy.shape(new_act[0:40,:])
	print numpy.shape(training_set[kl:40+kl,:])
	
	(ranks,correct,tr) = performIdentification(training_set[kl:40+kl,:],pred_act[kl:40+kl,:])
	print "Correct:", correct , "Mean rank:", numpy.mean(ranks)
	(ranks,correct,tr) = performIdentification(training_set[kl:40+kl,:],new_act[0:40,:])
	print "Correct:", correct , "Mean rank:", numpy.mean(ranks)

	
	ofs = run_nonlinearity_detection(numpy.mat(training_set),numpy.mat(pred_act))
	pred_act_t = apply_output_function(numpy.mat(pred_act),ofs)
	ofs = run_nonlinearity_detection(numpy.mat(training_set[kl:,:]),numpy.mat(new_act))
	new_act_t = apply_output_function(numpy.mat(new_act),ofs)


	(ranks,correct,tr) = performIdentification(training_set[kl:40+kl,:],pred_act_t[kl:40+kl,:])
	print "TFCorrect:", correct , "Mean tf_rank:", numpy.mean(ranks)
	(ranks,correct,tr) = performIdentification(training_set[kl:40+kl,:],new_act_t[0:40,:])
	print "TFCorrect:", correct , "Mean tf_rank:", numpy.mean(ranks)
	
	pylab.figure()
	pylab.hist(ranks)
	

def history_estim(x,hist,pred_act):
  		 (a,b,c,d) = list(x)
		 return numpy.multiply(a*(pred_act+3.0)+b,c*(hist+3.0)+d)
		 
def history_error(x,hist,pred_act,training_set):
		 return training_set - history_estim(x,hist,pred_act)
		 
def history_der(x,hist,pred_act,training_set):
		(a,b,c,d) = list(x)
		ad = numpy.multiply(hist , c*pred_act+d)
		bd = c*pred_act+d
		cd = numpy.multiply(a*hist+b,pred_act)
		dd = a*hist+b
		return numpy.vstack((ad,bd,cd,dd)).T	


def CompareRegressions():
	import scipy
	from scipy import linalg
	f = open("results.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[0]

	rfs  = node.children[0].data["ReversCorrelationRFs"]
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
	pred_val_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])
	pred_act_t  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities+TF"])
	pred_val_act_t  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities+TF"])
	training_set = node.data["training_set"]
	validation_set = node.data["validation_set"]
	#training_set = numpy.array(node.children[0].data["LaterTrainingSet"])
	#validation_set = numpy.array(node.children[0].data["LaterValidationSet"])
	#m = node.children[0].data["LaterModel"]
	
	
	
	training_inputs = node.data["training_inputs"]
	validation_inputs = node.data["validation_inputs"]
	raw_validation_set = node.data["raw_validation_set"]
	
	#for i in xrange(0,len(raw_validation_set)):
	#    raw_validation_set[i] = numpy.array(m.returnPredictedActivities(numpy.mat(raw_validation_set[i])))
	
	asd_rfs  = node.children[2].data["RFs"]
	
	
	
	# REGINV
	pylab.figure()
	m = numpy.max(numpy.abs(rfs))
        for i in xrange(0,numpy.shape(training_set)[1]):
            pylab.subplot(10,11,i+1)
            w = rfs[i]
            pylab.show._needmain=False
            pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	    pylab.axis('off')

	# ASD
	pylab.figure()	
	size = numpy.sqrt(numpy.shape(asd_rfs)[1])
	
        for i in xrange(0,numpy.shape(training_set)[1]):
	    m = numpy.max(numpy.abs(asd_rfs[i]))
            pylab.subplot(10,11,i+1)
            w = numpy.array(asd_rfs[i]).reshape(size,size)
            pylab.show._needmain=False
            pylab.imshow(w,vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	    pylab.axis('off')

	# create predicted activities
	
	asd_pred_act = numpy.array(numpy.mat(training_inputs) * numpy.mat(asd_rfs).T)
	asd_pred_val_act = numpy.array(numpy.mat(validation_inputs) * numpy.mat(asd_rfs).T)
	
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(asd_pred_act))
	asd_pred_act_t = apply_sigmoid_output_function(numpy.mat(asd_pred_act),ofs)
	asd_pred_val_act_t= apply_sigmoid_output_function(numpy.mat(asd_pred_val_act),ofs)

	
	raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
	
	signal_power,noise_power,normalized_noise_power,reg_training_prediction_power,reg_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, pred_act, pred_val_act)
	pylab.suptitle('Signal power estimation for Pseudo inverse. Averaged trials validation set.')
	print "Mean Reg. pseudoinverse POSITIVE prediction power on training set / validation set(averaged) :", numpy.mean(reg_training_prediction_power * (reg_training_prediction_power > 0)) , " / " , numpy.mean(reg_validation_prediction_power * (reg_validation_prediction_power > 0))
	
	
	signal_power,noise_power,normalized_noise_power,asd_training_prediction_power,asd_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, asd_pred_act, asd_pred_val_act)
	pylab.suptitle('Signal power estimation for ASDRD Averaged trials validation set.')
	print "Mean ASD POSITIVE prediction power on training set / validation set(averaged) :", numpy.mean(asd_training_prediction_power * (asd_training_prediction_power > 0)) , " / " , numpy.mean(asd_validation_prediction_power * (asd_validation_prediction_power > 0))
	
	pylab.figure()
	pylab.title('Before TF averaged trials')
	pylab.plot(reg_validation_prediction_power,asd_validation_prediction_power,'ro')
	pylab.plot([-2.0,2.0],[-2.0,2.0])
	pylab.axis([-2.0,2.0,-2.0,2.0])
	pylab.xlabel('Regurelized inverse prediction power')
	pylab.ylabel('ASDRD prediction power')

	signal_power,noise_power,normalized_noise_power,reg_training_prediction_power,reg_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, raw_validation_set[0], pred_act, pred_val_act)
	pylab.suptitle('Signal power estimation for Pseudo inverse. Single trial validation set.')
	print "Mean Reg. pseudoinverse POSITIVE prediction power on training set / validation set : ", numpy.mean(reg_training_prediction_power * (reg_training_prediction_power > 0)) , " / " , numpy.mean(reg_validation_prediction_power * (reg_validation_prediction_power > 0))
	
	signal_power,noise_power,normalized_noise_power,asd_training_prediction_power,asd_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, raw_validation_set[0], asd_pred_act, asd_pred_val_act)
	pylab.suptitle('Signal power estimation for ASDRD. Single trial validation set.')
	print "Mean ASD POSITIVE prediction power on training set / validation set : ", numpy.mean(asd_training_prediction_power * (asd_training_prediction_power > 0)) , " / " , numpy.mean(asd_validation_prediction_power * (asd_validation_prediction_power > 0))
	
	pylab.figure()
	pylab.title('Before TF single trial')
	pylab.plot(reg_validation_prediction_power,asd_validation_prediction_power,'ro')
	pylab.plot([-2.0,2.0],[-2.0,2.0])
	pylab.axis([-2.0,2.0,-2.0,2.0])
	pylab.xlabel('Regurelized inverse prediction power')
	pylab.ylabel('ASDRD prediction power')



	(ranks,correct,pred) = performIdentification(raw_validation_set[0],pred_val_act)
	print "Reg. pseudoinverse identification single trial> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(raw_validation_set[0] - pred_val_act,2)) 

	(ranks,correct,pred) = performIdentification(raw_validation_set[0],asd_pred_val_act)
	print "ASD identification single trial> Correct:", correct , "Mean rank:", numpy.mean(ranks),  "MSE", numpy.mean(numpy.power(raw_validation_set[0] - asd_pred_val_act,2)) 
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act)
	print "Reg. pseudoinverse identification trial average> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act,2))  

	(ranks,correct,pred) = performIdentification(validation_set,asd_pred_val_act)
	print "ASD identification trial average> Correct:", correct , "Mean rank:", numpy.mean(ranks), "MSE", numpy.mean(numpy.power(validation_set - asd_pred_val_act,2)) 
	
	

	# with transfer function analysis
	
	print '\n\n\nAFTER TRANSFER FUNCTION APPLICATION \n '
	
	signal_power,noise_power,normalized_noise_power,reg_training_prediction_power,reg_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, pred_act_t, pred_val_act_t)
	pylab.suptitle('Signal power estimation for Pseudo inverse. Averaged trials validation set with applied transfer function.')
	print "Mean Reg. pseudoinverse POSITIVE prediction power on training set / validation set(averaged): ", numpy.mean(reg_training_prediction_power * (reg_training_prediction_power > 0)) , " / " , numpy.mean(reg_validation_prediction_power * (reg_validation_prediction_power > 0))
	
	signal_power,noise_power,normalized_noise_power,asd_training_prediction_power,asd_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, validation_set, asd_pred_act_t, asd_pred_val_act_t)
	pylab.suptitle('Signal power estimation for ASDRD. Averaged trials validation set with applied transfer function.')
	print "Mean ASD POSITIVE prediction power on training set / validation set(averaged): ", numpy.mean(asd_training_prediction_power * (asd_training_prediction_power > 0)) , " / " , numpy.mean(asd_validation_prediction_power * (asd_validation_prediction_power > 0))

	pylab.figure()
	pylab.title('After TF averaged trials')
	pylab.plot(reg_validation_prediction_power,asd_validation_prediction_power,'ro')
	pylab.plot([-2.0,2.0],[-2.0,2.0])
	pylab.axis([-2.0,2.0,-2.0,2.0])
	pylab.xlabel('Regurelized inverse prediction power')
	pylab.ylabel('ASDRD prediction power')


	signal_power,noise_power,normalized_noise_power,reg_training_prediction_power,reg_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, raw_validation_set[0], pred_act_t, pred_val_act_t)
	pylab.suptitle('Signal power estimation for Pseudo inverse. Single trial validation set with applied transfer function.')
	print "Mean Reg. pseudoinverse POSITIVE prediction power on training set / validation set: ", numpy.mean(reg_training_prediction_power * (reg_training_prediction_power > 0)) , " / " , numpy.mean(reg_validation_prediction_power * (reg_validation_prediction_power > 0))
	
	signal_power,noise_power,normalized_noise_power,asd_training_prediction_power,asd_validation_prediction_power = signal_power_test(raw_validation_data_set, training_set, raw_validation_set[0], asd_pred_act_t, asd_pred_val_act_t)
	pylab.suptitle('Signal power estimation for ASDRD. Single trial validation set with applied transfer function.')
	print "Mean ASD POSITIVE prediction power on training set / validation set: ", numpy.mean(asd_training_prediction_power * (asd_training_prediction_power > 0)) , " / " , numpy.mean(asd_validation_prediction_power * (asd_validation_prediction_power > 0))

	pylab.figure()
	pylab.title('After TF single trial')
	pylab.plot(reg_validation_prediction_power,asd_validation_prediction_power,'ro')
	pylab.plot([-2.0,2.0],[-2.0,2.0])
	pylab.axis([-2.0,2.0,-2.0,2.0])
	pylab.xlabel('Regurelized inverse prediction power')
	pylab.ylabel('ASDRD prediction power')


	(ranks,correct,pred) = performIdentification(raw_validation_set[0],pred_val_act_t)
	print "Reg. pseudoinverse identification single trial> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(raw_validation_set[0] - pred_val_act_t,2)) 

	(ranks,correct,pred) = performIdentification(raw_validation_set[0],asd_pred_val_act_t)
	print "ASD identification single trial> Correct:", correct , "Mean rank:", numpy.mean(ranks),  "MSE", numpy.mean(numpy.power(raw_validation_set[0] - asd_pred_val_act_t,2)) 
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act_t)
	print "Reg. pseudoinverse identification trial average> Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act_t,2))  

	(ranks,correct,pred) = performIdentification(validation_set,asd_pred_val_act_t)
	print "ASD identification trial average> Correct:", correct , "Mean rank:", numpy.mean(ranks), "MSE", numpy.mean(numpy.power(validation_set - asd_pred_val_act_t,2)) 
	
	pylab.show()
	
	
def DeepLook():
	import scipy
	from scipy import linalg
	f = open("results.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[0]

	rfs  = node.children[0].data["ReversCorrelationRFs"]
	sx,sy = numpy.shape(rfs[0])
	asd_rfs  = node.children[1].data["RFs"]
	asdrd_rfs  = node.children[2].data["RFs"]
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
	pred_val_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])
	
	training_set_old = node.data["training_set"]
	validation_set_old = node.data["validation_set"]
	training_set = numpy.array(node.children[0].data["LaterTrainingSet"])
	validation_set = numpy.array(node.children[0].data["LaterValidationSet"])
	training_inputs = node.data["training_inputs"]
	validation_inputs = node.data["validation_inputs"]
	
	print "Mean of old one:",numpy.mean(training_set_old) , " Variance of old one:", numpy.mean(numpy.var(training_set_old,axis=0))
	print "Mean of modified:",numpy.mean(training_set) , " Variance of modified:", numpy.mean(numpy.var(training_set,axis=0))
	print "Mean of predicted:",numpy.mean(pred_act) , " Variance of predicted:", numpy.mean(numpy.var(pred_act,axis=0))
	
	
	#(e,te,c,tc,RFs,pred_act,pred_val_act,corr_coef,corr_coef_tf) = regulerized_inverse_rf(training_inputs,training_set,sx,sy,__main__.__dict__.get('Alpha',50),numpy.mat(validation_inputs),validation_set,contrib.dd.DB2(None),True)
    
	
	valdataset = loadSimpleDataSet("Mice/2009_11_04/region3_50stim_10reps_15fr_103cells_on_response_spikes",50,103,10)
	validation_inputs_big=generateInputs(valdataset,"/home/antolikjan/topographica/topographica/Mice/2009_11_04/","/20091104_50stimsequence/50stim%04d.tif",__main__.__dict__.get('density', 0.4),1.8,offset=0)
	
	
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act),display=False)
	pred_act_t = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
	pred_val_act_t= apply_sigmoid_output_function(numpy.mat(pred_val_act),ofs)

	val_errors = numpy.array(numpy.power(pred_val_act - validation_set,2))
	val_errors_t = numpy.array(numpy.power(pred_val_act_t - validation_set,2))
	neurons = [0,1,8,15,17,19,24,25,27,33,37,38]#,40,42,44,45,46,47,48,53,55,56,58,61,65,65,75,85,93,96]	
	
	
	ssy,ssx = numpy.shape(validation_inputs_big[0])
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act)
	print "Without TF. Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act,2))
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act_t)
	print "With TF. Correct:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act_t,2))
	
	
	for n in neurons:
	    s = numpy.argsort(val_errors_t[:,n])[::-1]
	    f = pylab.figure()
	    tn = 8 
	    
	    ax = f.add_axes([0.01,0.75,1.0/(tn+1)-0.02,0.24])
	    m = numpy.max([-numpy.min(rfs[n]),numpy.max(rfs[n])])
            pylab.imshow(rfs[n],vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	    
	    ax = f.add_axes([0.01,0.5,1.0/(tn+1)-0.02,0.24])
	    m = numpy.max([-numpy.min(asd_rfs[n]),numpy.max(asd_rfs[n])])
            pylab.imshow(numpy.reshape(asd_rfs[n],(sx,sy)),vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	    
	    ax = f.add_axes([0.01,0.25,1.0/(tn+1)-0.02,0.24])
	    m = numpy.max([-numpy.min(asdrd_rfs[n]),numpy.max(asdrd_rfs[n])])
            pylab.imshow(numpy.reshape(asdrd_rfs[n],(sx,sy)),vmin=-m,vmax=m,interpolation='nearest',cmap=pylab.cm.RdBu)
	    
	    m = numpy.max([-numpy.min(rfs[n]),numpy.max(rfs[n])])
	    for i in xrange(0,tn):
		ax = f.add_axes()
		
		ax = f.add_axes([(i+1.0)/(tn+1.0),0.75,1.0/(tn+1)-0.02,0.24])
  	        ax.imshow(validation_inputs_big[s[i]],vmin=0,vmax=256,interpolation='nearest',cmap=pylab.cm.gray)
		ax.axis('off')
		ax.add_line(matplotlib.lines.Line2D([ssx*0.3,ssx*0.3],[ssy*0.0,ssy*1.0]))
		ax.add_line(matplotlib.lines.Line2D([ssx*0.8,ssx*0.8],[ssy*0.0,ssy*1.0]))
			
		ax = f.add_axes([(i+1.0)/(tn+1.0),0.5,1.0/(tn+1)-0.02,0.24])
  	        ax.imshow(numpy.multiply(numpy.reshape(validation_inputs[s[i]],(sx,sy)),numpy.abs(rfs[n])/m),vmin=-135,vmax=135,interpolation='nearest',cmap=pylab.cm.gray)
		ax.axis('off')
			
		ax = f.add_axes([(i+1.0)/(tn+1.0),0.25,1.0/(tn+1)-0.02,0.15])
  	        ax.plot(validation_set[:,n],'ro')
		ax.plot(pred_val_act_t[:,n],'bo')
		ax.axvline(s[i])
		
		ax = f.add_axes([(i+1.0)/(tn+1.0),0.05,1.0/(tn+1)-0.02,0.15])
  	        ax.plot(numpy.mean(validation_set,axis=1),'ro')
		ax.axvline(s[i])



def SuperModel():
	import scipy
	from scipy import linalg
	f = open("results.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[0]

	rfs  = node.children[0].data["ReversCorrelationRFs"][0:103]
	#fitted_rfs  = node.children[0].data["FittedRFs"][0:103]
	
	#SurrRFs = node.children[0].children[0].children[4].data["SurrRFs"]
	#SurrTI = node.children[0].children[0].data["TrainingInputs"]
	#SurrVI = node.children[0].children[0].data["ValidationInputs"]

	
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"][:,0:103])
	pred_val_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"][:,0:103])
	
	training_set = node.data["training_set"][:,0:103]
	validation_set = node.data["validation_set"][:,0:103]
	
	#training_set = numpy.array(node.children[0].data["LaterTrainingSet"])
	#validation_set = numpy.array(node.children[0].data["LaterValidationSet"])

	training_inputs = node.data["training_inputs"]
	validation_inputs = node.data["validation_inputs"]
	raw_validation_set = node.data["raw_validation_set"]
	
	rf_mag = [numpy.sum(numpy.power(r,2)) for r in rfs]
	
	#discard ugly RFs          	
	pylab.figure()
	pylab.hist(rf_mag)
	#pylab.show()
	
	to_delete = numpy.nonzero((numpy.array(rf_mag) < 0.000000)*1.0)[0]
	print to_delete
	rfs = numpy.delete(rfs,to_delete,axis=0)
	pred_act = numpy.delete(pred_act,to_delete,axis=1)
	pred_val_act = numpy.delete(pred_val_act,to_delete,axis=1)
	training_set = numpy.delete(training_set,to_delete,axis=1)
	validation_set = numpy.delete(validation_set,to_delete,axis=1)
	
	#for i in xrange(0,len(raw_validation_set)):
	#    raw_validation_set[i] = numpy.delete(raw_validation_set[i],to_delete,axis=1)
	
	
	(sx,sy) = numpy.shape(rfs[0])
	#pred_act  =  numpy.array(numpy.mat(training_inputs)*numpy.mat(numpy.reshape(fitted_rfs,(103,sx*sy)).T))
	#pred_val_act  =  numpy.array(numpy.mat(validation_inputs)*numpy.mat(numpy.reshape(fitted_rfs,(103,sx*sy)).T))
	
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act))
	pred_act_t = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
	pred_val_act_t= apply_sigmoid_output_function(numpy.mat(pred_val_act),ofs)
	
	c=[]
	for i in xrange(0,103-len(to_delete)):
		c.append(numpy.corrcoef(validation_set[:,i],pred_val_act_t[:,i])[0][1])
	print c
	print numpy.mean(c)
	#return
	
	#z = numpy.argsort(numpy.mean(training_set,axis=0))
	#pylab.figure()
	#pylab.plot(numpy.mean(training_set[:,z[0:10]],axis=1),numpy.mean(training_set[:,z[80:103]],axis=1),'ro')
	#pylab.figure()
	#pylab.plot(numpy.mean(pred_act[:,z[0:10]],axis=1),numpy.mean(pred_act[:,z[80:103]],axis=1),'ro')
	#pylab.figure()
	#pylab.plot(numpy.mean(pred_act_t[:,z[0:10]],axis=1),numpy.mean(pred_act_t[:,z[80:103]],axis=1),'ro')

	
	
	#temp = numpy.reshape(rfs,(1800,sx*sy))
	
	#var = numpy.mat(training_inputs) * numpy.mat(numpy.abs(temp))
	#val_var = numpy.mat(validation_inputs) * numpy.mat(temp)
	
	#var = numpy.var(training_inputs,axis=1)
	#val_var = numpy.var(validation_inputs,axis=1)
	#dataset= loadSimpleDataSet('/home/antolikjan/topographica/topographica/Mice/20090925_14_36_01/spont_filtered.dat',2852,50,num_rep=1,num_frames=1,offset=0,transpose=False)
	#spont = generateTrainingSet(dataset)
			
	spont_corr,p = pearcorr(training_set)
	
	print numpy.shape(training_set)
	print numpy.shape(spont_corr)
	print numpy.shape(numpy.eye(len(rfs)))
	spont_corr = numpy.multiply(numpy.multiply(spont_corr,abs(numpy.eye(len(rfs))-1.0)),(p<0.000001)*1.0)
	
	pylab.figure()
	pylab.imshow(spont_corr)
	pylab.colorbar()
	
	
	
	#var1=var2=var3 = numpy.array(numpy.mat(training_set)*numpy.mat(spont_corr)) 
	var1=var2=var3 = numpy.array(node.children[3].data["ReversCorrelationPredictedActivities+TF"][:,0:103])
	
	#training_set = training_set-numpy.array(numpy.mat(training_set)*numpy.mat(spont_corr))
	#validation_set = validation_set-numpy.array(numpy.mat(validation_set)*numpy.mat(spont_corr))
	
	#val_var = numpy.zeros(numpy.shape(validation_set))
	#val_var1 = val_var2 = val_var3=  numpy.array(numpy.mat(validation_set)*numpy.mat(spont_corr))
	val_var1 = val_var2 = val_var3 = numpy.array(node.children[3].data["ReversCorrelationPredictedValidationActivities+TF"][:,0:103])
	
	#var  =  numpy.mat(SurrTI)*numpy.mat(numpy.reshape(SurrRFs,(103,sx*sy)).T)
	#val_var  = numpy.mat(SurrVI)*numpy.mat(numpy.reshape(SurrRFs,(103,sx*sy)).T)
	
	#try what effect rotated RFs could have
	if False: 
		rfs_90 = []
		rfs_180 = []
		rfs_270 = []
		for rf in rfs:
			r90 = rot90_around_center_of_gravity(rf)
			#r180 = rot90_around_center_of_gravity(r90)
			#r270 = rot90_around_center_of_gravity(r180)
			r180 = rf*-1.0
			r270 = r90*-1.0
			rfs_90.append(r90.flatten())	
			rfs_180.append(r180.flatten())
			rfs_270.append(r270.flatten())
			var1  =  numpy.mat(training_inputs)*numpy.mat(rfs_90).T
			val_var1  = numpy.mat(validation_inputs)*numpy.mat(rfs_90).T
			var2  =  numpy.mat(training_inputs)*numpy.mat(rfs_180).T
			val_var2  = numpy.mat(validation_inputs)*numpy.mat(rfs_180).T
			var3  =  numpy.mat(training_inputs)*numpy.mat(rfs_270).T
			val_var3  = numpy.mat(validation_inputs)*numpy.mat(rfs_270).T
	
	
	
	from scipy.optimize import leastsq
	xs = []
	err = []
	for i in xrange(0,len(rfs)):
	    #print i
	    min_err = 100000000000000000
	    xo=True
	    for r in xrange(0,10):
		rand =numbergen.UniformRandom(seed=513)
		r0 = (numpy.array([rand(),rand(),rand(),rand(),rand()])-0.5)*2.0
		x0 = [0.0,0.0,0.0,0.0,1.0]
		rand_scale = [3.0,3.0,3.0,3.0,3.0]
		x0 = x0 + numpy.multiply(r0,rand_scale)
		
		xopt = leastsq(supermodel_error, x0[:], args=(numpy.array(var1)[:,i],numpy.array(var2)[:,i],numpy.array(var3)[:,i],numpy.array(pred_act)[:,i],numpy.array(training_set)[:,i]),ftol=0.0000000000000000001,xtol=0.0000000000000001,warning=False)
		
		er = numpy.sum(supermodel_error(xopt[0],numpy.array(var1)[:,i],numpy.array(var2)[:,i],numpy.array(var3)[:,i],numpy.array(pred_act)[:,i],numpy.array(training_set)[:,i])**2)
		if min_err > er:
		   min_err = er	
	    	   xo=xopt[0] 
				
	    xs.append(xo)	
	    new_error  = numpy.sum(supermodel_error(xo,numpy.array(var1)[:,i],numpy.array(var2)[:,i],numpy.array(var3)[:,i],numpy.array(pred_act)[:,i],numpy.array(training_set)[:,i])**2)
	    old_error =  numpy.sum((numpy.array(pred_act)[:,i] - numpy.array(training_set)[:,i])**2)
	    err.append( (old_error - new_error)/old_error * 100)
	
	
	print numpy.mat(xs)
	
	print "Training error decreased by:", numpy.mean(err) , '%'
	new_val_act = apply_supermodel_estim(xs,val_var1,val_var2,val_var3,pred_val_act)
	new_act = apply_supermodel_estim(xs,var1,var2,var3,pred_act)
	
	#new_act = pred_act
	#new_val_act = pred_val_act
	
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(new_act))
	new_val_act_t= numpy.array(apply_sigmoid_output_function(numpy.mat(new_val_act),ofs))
	new_act_t= numpy.array(apply_sigmoid_output_function(numpy.mat(new_act),ofs))
	
	pylab.figure()
	for i in xrange(0,103):
		pylab.subplot(11,11,i+1)    
	    	pylab.plot(pred_val_act[:,i],validation_set[:,i],'o')
	

	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act)
	print "Original:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act,2))
	
	(ranks,correct,pred) = performIdentification(validation_set,pred_val_act_t)
	print "TF+Original:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act_t,2))
	
	#validation_set = validation_set-numpy.array(numpy.mat(validation_set)*numpy.mat(spont_corr))*numpy.array(numpy.tile(numpy.mat(xs)[:,4].T,(len(validation_set),1)))
	
	#(e,te,c,tc,RFs,pred_act,pred_val_act,corr_coef,corr_coef_tf) = regulerized_inverse_rf(numpy.array(training_inputs),numpy.array(training_set),sx,sy,__main__.__dict__.get('Alpha',50),numpy.mat(validation_inputs),numpy.mat(validation_set),contrib.dd.DB(None),True)
	
	#examine_correlated_noise(validation_set,raw_validation_set,xs,pred_val_act,spont_corr)
	
	(ranks,correct,pred) = performIdentification(validation_set,new_val_act)
	print "After contrast normalization:", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - new_val_act,2))
	
	(ranks,correct,pred) = performIdentification(validation_set,new_val_act_t)
	print "After contrast normalization:+TF", correct , "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - new_val_act_t,2))
	
	raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
		
	print 'ORIGINAL:'
	
	signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), pred_act, pred_val_act)
	signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), pred_act_t, pred_val_act_t)
	
	print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power*(training_prediction_power>0)) , " / " , numpy.mean(validation_prediction_power)
	print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(validation_prediction_power_t)

	print 'NEW:'
	signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), new_act, new_val_act)
	signal_power,noise_power,normalized_noise_power,training_prediction_power_t,validation_prediction_power_t = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), new_act_t, new_val_act_t)
	
	print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power*(training_prediction_power>0)) , " / " , numpy.mean(validation_prediction_power)
	print "Prediction power after TF on training set / validation set: ", numpy.mean(training_prediction_power_t) , " / " , numpy.mean(validation_prediction_power_t)



def supermodel_estim(x,var1,var2,var3,pred_act):
  		 (a,b,c,d,e) = list(x)
		 #return numpy.divide(a*pred_act+b,c*var+d)
		 #zz=  (a*numpy.max([numpy.zeros(numpy.shape(pred_act)),pred_act+b],axis=0) + c*numpy.max([numpy.zeros(numpy.shape(pred_act)),-pred_act+d],axis=0))+e*var
		 #zz=  numpy.divide(a*pred_act,1.0 + numpy.max([numpy.zeros(numpy.shape(pred_act)),e*numpy.abs(var1)+b]))
		 #zz = numpy.divide(a*pred_act,numpy.max([numpy.ones(numpy.shape(pred_act)),e + b*var1 + c*var2 + d*var3],axis=0))
		 #zz = numpy.multiply(a*pred_act,1.0+ d*numpy.abs(var1)+b)
		 zz = a*pred_act + e*var1 
		 #zz = 1.0*var1
		 #zz =  a*(numpy.shape(pred_act)+b)*var
		 #return numpy.divide(zz+e,f*var+g)
		 return zz
		 
def supermodel_error(x,var1,var2,var3,pred_act,training_set):
		 return training_set - supermodel_estim(x,var1,var2,var3,pred_act)


def apply_supermodel_estim(params,var1,var2,var3,pred_act):
    new_act = []
    for i in xrange(0,len(params)):
	 new_act.append(supermodel_estim(params[i],numpy.array(var1)[:,i],numpy.array(var2)[:,i],numpy.array(var3)[:,i],numpy.array(pred_act)[:,i]))
    return numpy.array(numpy.mat(new_act).T)

def examine_correlated_noise(validation_set,raw_validation_set,params,pred_val_act,spont_corr):
    
    raw_later_act=[]
    for raw in raw_validation_set:
	var=numpy.mat(raw)*numpy.mat(spont_corr)  
    	raw_later_act.append(var) 
	
    # MSE with predictions from the same trials
    m=[]
    m_orig=[]
    for (rawlat,rawvs) in zip(raw_later_act,raw_validation_set):
	m.append(MSE(apply_supermodel_estim(params,rawlat,rawlat,rawlat,pred_val_act),rawvs))
	m_orig.append(MSE(pred_val_act,rawvs))
    
    print "MSE from matching trials, ORIGINAL:",numpy.mean(m_orig)
    print "MSE from averaged trials, ORIGINAL:",MSE(pred_val_act,numpy.mean(numpy.array(raw_validation_set),0))	
    print "MSE from matching trials:",numpy.mean(m)
    var=numpy.mat(numpy.mean(numpy.array(raw_validation_set),0))*numpy.mat(spont_corr)
    print "MSE from averaged trials:",MSE(apply_supermodel_estim(params,var,var,var,pred_val_act),numpy.mean(numpy.array(raw_validation_set),0))
    
    # MSE with predictions from the same trials
    m=[]
    m_orig=[]	
    for j in xrange(0,len(raw_later_act)):
	for k in xrange(0,len(raw_later_act)):
	    if j != k:
	       m.append(MSE(apply_supermodel_estim(params,raw_later_act[k],raw_later_act[k],raw_later_act[k],pred_val_act),raw_validation_set[j]))    
    print "MSE from different trials",numpy.mean(m)	
	
def MSE(predictions,targets):
    return numpy.mean(numpy.power(predictions-targets,2))	
	
	

def spontaneousActivity():
	import scipy
	import scipy.stats
	from scipy import linalg
	f = open("results.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[9]
	rfs = node.children[0].data["ReversCorrelationRFs"]
	
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
	val_pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])
	training_set = node.data["training_set"]
	validation_set = node.data["validation_set"]
	#flat_validation_set = node.data["flat_validation_set"]
	#flat_validation_inputs = node.data["flat_validation_inputs"]
	
	(z,sizex,sizey) = numpy.shape(rfs)
	
	#flat_val_pred_act = numpy.mat(flat_validation_inputs) *numpy.mat(numpy.reshape(rfs,(z,size*sizey))).T
	
	#dataset = loadSimpleDataSet('/home/antolikjan/topographica/topographica/Mice/20091110_19_16_53/spont_filtered.dat',5952,68,num_rep=1,num_frames=1,offset=0,transpose=False)
	dataset= loadSimpleDataSet('/home/antolikjan/topographica/topographica/Mice/20090925_14_36_01/spont_filtered.dat',2852,50,num_rep=1,num_frames=1,offset=0,transpose=False)
	spont = generateTrainingSet(dataset)		
	
	f = file("./Mice/20090925_14_36_01/(20090925_14_36_01)-_retinotopy_region2_sequence_50cells_cell_locations.txt", "r")
    	loc= [line.split() for line in f]
	(a,b) = numpy.shape(loc)
	for i in xrange(0,a):
		for j in xrange(0,b):
			loc[i][j] = float(loc[i][j])

	
	
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act))
	pred_act_t = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
	val_pred_act_t = apply_sigmoid_output_function(numpy.mat(val_pred_act),ofs)

	diff_act = pred_act - training_set
	diff_act_t = pred_act_t - training_set
	val_diff_act = val_pred_act - validation_set
	val_diff_act_t = val_pred_act_t - validation_set

	
	rfs_corr=[]
	spont_corr=[]
	diff_corr=[]
	diff_t_corr=[]
	act_corr=[]
	val_act_corr=[]
	val_diff_corr=[]
	val_diff_t_corr=[]
	pred_val_act_t_corr=[]

	for i in xrange(0,len(rfs)):		
	    for j in xrange(i+1,len(rfs)):
		rfs_corr.append(scipy.stats.pearsonr(rfs[i].flatten(), rfs[j].flatten())[0])		
		spont_corr.append(scipy.stats.pearsonr(spont[:,i], spont[:,j])[0])
		diff_corr.append(scipy.stats.pearsonr(diff_act[:,i], diff_act[:,j])[0])
		diff_t_corr.append(scipy.stats.pearsonr(diff_act_t[:,i], diff_act[:,j])[0])
		act_corr.append(scipy.stats.pearsonr(training_set[:,i], training_set[:,j])[0])
		val_act_corr.append(scipy.stats.pearsonr(validation_set[:,i], validation_set[:,j])[0])
		val_diff_corr.append(scipy.stats.pearsonr(val_diff_act[:,i], val_diff_act[:,j])[0])
		val_diff_t_corr.append(scipy.stats.pearsonr(val_diff_act_t[:,i], val_diff_act[:,j])[0])
		pred_val_act_t_corr.append(scipy.stats.pearsonr(val_pred_act_t[:,i], val_pred_act_t[:,j])[0])
		
				
	pylab.figure()
	pylab.title('Correlation between spontaneous activity correlations and RFs correlations')
	pylab.plot(rfs_corr,spont_corr,'bo')
	pylab.xlabel('RFs correlations')
	pylab.ylabel('Spontaneous activity correlations')
	
	pylab.figure()
	pylab.title('Correlation between triggered activity correlations and RFs correlations')
	pylab.plot(act_corr,spont_corr,'bo')
	pylab.xlabel('Triggered activity')
	pylab.ylabel('Spontaneous activity correlations')
	
	
	pylab.figure()
	pylab.title('Correlation between spontaneous activity correlations and prediction residuals correlations after removal of nonspecific RFs')
	pylab.plot(diff_corr,spont_corr,'bo')
	pylab.xlabel('Residuals correlations')
	pylab.ylabel('Spontaneous activity correlations')
	
	pylab.figure()
	pylab.title('Correlation between spontaneous activity correlations and prediction residuals +TF correlations after removal of nonspecific RFs')
	pylab.plot(diff_t_corr,spont_corr,'bo')
	pylab.xlabel('Residuals correlations+TF')
	pylab.ylabel('Spontaneous activity correlations')
	
	pylab.figure()
	pylab.title('Correlation between triggered activity correlations and prediction residuals correlations after removal of nonspecific RFs')
	pylab.plot(act_corr,diff_corr,'bo')
	pylab.xlabel('Triggered activity')
	pylab.ylabel('Residuals correlations')
	
	
        print 'Correlation between RFs corr. and Spont act corr.:', scipy.stats.pearsonr(rfs_corr,spont_corr)
	
	print 'On training set:'
	print 'Correlation between Triggered activity corr. and Spont act corr.:', scipy.stats.pearsonr(act_corr,spont_corr)
	print 'Correlation between Residuals corr. and Spont act corr.:', scipy.stats.pearsonr(diff_corr,spont_corr)
	print 'Correlation between Residuals+TF corr. and Spont act corr.:', scipy.stats.pearsonr(diff_t_corr,spont_corr)
	print 'Correlation between Residuals corr. and Triggered act corr.:', scipy.stats.pearsonr(diff_corr,act_corr)
        	
	print 'On validation set:'		
	print 'Correlation between Triggered validation activity corr. and Spont act corr.:', scipy.stats.pearsonr(val_act_corr,spont_corr)
        print 'Correlation between Residuals corr. and Spont act corr.:', scipy.stats.pearsonr(val_diff_corr,spont_corr)
	print 'Correlation between Residuals+TF corr. and Spont act corr.:', scipy.stats.pearsonr(val_diff_t_corr,spont_corr)
	print 'Correlation between Residuals corr. and Triggered act corr.:', scipy.stats.pearsonr(val_diff_corr,val_act_corr)
	print 'Correlation between predicted validation activities corr. and Spont act corr.:', scipy.stats.pearsonr(pred_val_act_t_corr,spont_corr)
	print 'Correlation between difference of predicted validation activities and measured validation activities corr. and Spont act corr.:', scipy.stats.pearsonr(numpy.array(val_act_corr)-numpy.array(pred_val_act_t_corr),spont_corr)
	
	
	#remove ugly neurons
	print 'Removing weak RFs'
	rfs_mag=numpy.sum(numpy.reshape(numpy.abs(numpy.array(rfs)),(len(rfs),numpy.size(rfs[0]))),axis=1)
	to_delete = numpy.nonzero((rfs_mag < 0.03) * 1.0)
	pylab.figure()
	pylab.hist(rfs_mag)
	
	rfs  = numpy.delete(numpy.array(rfs),to_delete[0],axis=0)	
	spont  = numpy.array(numpy.delete(numpy.mat(spont),to_delete[0],axis=1))
	diff_act  = numpy.array(numpy.delete(numpy.mat(diff_act),to_delete[0],axis=1))
	diff_act_t  = numpy.array(numpy.delete(numpy.mat(diff_act_t),to_delete[0],axis=1))
	training_set  = numpy.array(numpy.delete(numpy.mat(training_set),to_delete[0],axis=1))
	validation_set  = numpy.array(numpy.delete(numpy.mat(validation_set),to_delete[0],axis=1))
	val_diff_act  = numpy.array(numpy.delete(numpy.mat(val_diff_act),to_delete[0],axis=1))
	val_diff_act_t  = numpy.array(numpy.delete(numpy.mat(val_diff_act_t),to_delete[0],axis=1))
	val_pred_act_t  = numpy.array(numpy.delete(numpy.mat(val_pred_act_t),to_delete[0],axis=1))
	
	spont_corr=[]
	diff_corr=[]
	diff_t_corr=[]
	act_corr=[]
	val_act_corr=[]
	val_diff_corr=[]
	val_diff_t_corr=[]
	pred_val_act_t_corr=[]
	
	pos_rfs_corr=[]
	pos_spont_corr=[]
	pos_dist=[]
	neg_rfs_corr=[]
	neg_spont_corr=[]
	neg_dist=[]
	
	for i in xrange(0,len(rfs)):		
	    for j in xrange(i+1,len(rfs)):
		cor = numpy.corrcoef(rfs[i].flatten(), rfs[j].flatten())[0][1]
		if cor > 0:
			pos_rfs_corr.append(cor)		
			pos_spont_corr.append(numpy.corrcoef(spont[:,i], spont[:,j])[0])
			pos_dist.append(distance(loc,i,j))
			
	        else:
			neg_rfs_corr.append(cor)		
			neg_spont_corr.append(numpy.corrcoef(spont[:,i], spont[:,j])[0])
			neg_dist.append(distance(loc,i,j))
		spont_corr.append(scipy.stats.pearsonr(spont[:,i], spont[:,j])[0])
		diff_corr.append(scipy.stats.pearsonr(diff_act[:,i], diff_act[:,j])[0])
		diff_t_corr.append(scipy.stats.pearsonr(diff_act_t[:,i], diff_act[:,j])[0])
		act_corr.append(scipy.stats.pearsonr(training_set[:,i], training_set[:,j])[0])
		val_act_corr.append(scipy.stats.pearsonr(validation_set[:,i], validation_set[:,j])[0])
		val_diff_corr.append(scipy.stats.pearsonr(val_diff_act[:,i], val_diff_act[:,j])[0])
		val_diff_t_corr.append(scipy.stats.pearsonr(val_diff_act_t[:,i], val_diff_act[:,j])[0])
		pred_val_act_t_corr.append(scipy.stats.pearsonr(val_pred_act_t[:,i], val_pred_act_t[:,j])[0])
	
	pylab.figure()
	pylab.title('Correlation between spontaneous activit correlations and RFs positive correlations after removal of nonspecific RFs')
	pylab.plot(pos_rfs_corr,pos_spont_corr,'bo')
	pylab.xlabel('RFs correlations')
	pylab.ylabel('Spontaneous activity correlations')
	
	

	
	#fig = pylab.figure()
	#from mpl_toolkits.mplot3d import Axes3D
	#ax = Axes3D(fig)
	#pylab.title('Correlation between spontaneous activit correlations and RFs positive correlations after removal of nonspecific RFs')
	#ax.scatter(pos_rfs_corr,pos_spont_corr,pos_dist)
	#ax.set_xlabel('RFs correlations')
	#ax.set_ylabel('Spontaneous activity correlations')
	#ax.set_zlabel('distance')

	
	pylab.figure()
	pylab.title('Correlation between spontaneous activit correlations and RFs negative correlations after removal of nonspecific RFs')
	pylab.plot(neg_rfs_corr,neg_spont_corr,'bo')
	pylab.xlabel('RFs correlations')
	pylab.ylabel('Spontaneous activity correlations')
	
	
	#fig = pylab.figure()
	#from mpl_toolkits.mplot3d import Axes3D
	#ax = Axes3D(fig)
	#pylab.title('Correlation between spontaneous activit correlations and RFs positive correlations after removal of nonspecific RFs')
	#ax.scatter(neg_rfs_corr,neg_spont_corr,neg_dist)
	#ax.set_xlabel('RFs correlations')
	#ax.set_ylabel('Spontaneous activity correlations')
	#ax.set_zlabel('distance')
	
	print 'Correlation between positive RFs corr. and Spont act corr.:', numpy.corrcoef(pos_rfs_corr,pos_spont_corr)
	print 'Correlation between negative RFs corr. and Spont act corr.:', numpy.corrcoef(neg_rfs_corr,neg_spont_corr)[0][1]
        
	print 'On training set:'
	print 'Correlation between Triggered activity corr. and Spont act corr.:', numpy.corrcoef(act_corr,spont_corr)
	print 'Correlation between Residuals corr. and Spont act corr.:', numpy.corrcoef(diff_corr,spont_corr)
	print 'Correlation between Residuals+TF corr. and Spont act corr.:', numpy.corrcoef(diff_t_corr,spont_corr)
	print 'Correlation between Residuals corr. and Triggered act corr.:', numpy.corrcoef(diff_corr,act_corr)
        	
	print 'On validation set:'		
	print 'Correlation between Triggered validation activity corr. and Spont act corr.:', numpy.corrcoef(val_act_corr,spont_corr)		
        print 'Correlation between Residuals corr. and Spont act corr.:', numpy.corrcoef(val_diff_corr,spont_corr)
	print 'Correlation between Residuals+TF corr. and Spont act corr.:', numpy.corrcoef(val_diff_t_corr,spont_corr)
	print 'Correlation between Residuals corr. and Triggered act corr.:', numpy.corrcoef(val_diff_corr,val_act_corr)
       	print 'Correlation between predicted validation activities corr. and Spont act corr.:', numpy.corrcoef(pred_val_act_t_corr,spont_corr)
	print 'Correlation between difference of predicted validation activities and measured validation activities corr. and Spont act corr.:', numpy.corrcoef(numpy.array(val_act_corr)-numpy.array(pred_val_act_t_corr),spont_corr)
		


def RFestimationFromOtherCells():
	import scipy
	from scipy import linalg
	f = open("results.dat",'rb')
	import pickle
	dd = pickle.load(f)
	node = dd.children[0]
	rfs = node.children[0].data["ReversCorrelationRFs"]
	pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedActivities"])
	val_pred_act  = numpy.array(node.children[0].data["ReversCorrelationPredictedValidationActivities"])

	training_set = node.data["training_set"]
	validation_set = node.data["validation_set"]
	training_inputs = node.data["training_inputs"]
	validation_inputs = node.data["validation_inputs"]
	raw_validation_set = node.data["raw_validation_set"]
	
	ofs = fit_sigmoids_to_of(numpy.mat(training_set),numpy.mat(pred_act))
	pred_act_t = apply_sigmoid_output_function(numpy.mat(pred_act),ofs)
	val_pred_act_t = apply_sigmoid_output_function(numpy.mat(val_pred_act),ofs)

	(later_pred_act,later_pred_val_act) = later_interaction_prediction(training_set,pred_act_t,validation_set,val_pred_act_t,raw_validation_set,node.children[0])
	
	
	#f = open("results.dat",'wb')
    	#pickle.dump(dd,f,-2)
	#f.close()
	
	return
 
	(z,sizex,sizey) = numpy.shape(rfs)
	
	dataset= loadSimpleDataSet('/home/antolikjan/topographica/topographica/Mice/20090925_14_36_01/spont_filtered.dat',2852,50,num_rep=1,num_frames=1,offset=0,transpose=False)
	spont = generateTrainingSet(dataset)		
	
	
	trig_corr,p = pearcorr(training_set)
	trig_corr = numpy.multiply(numpy.multiply(trig_corr,abs(numpy.eye(z)-1.0)),(p<0.01)*1.0)
	
	spont_corr,p = pearcorr(spont)
	spont_corr = numpy.multiply(numpy.multiply(spont_corr,abs(numpy.eye(z)-1.0)),(p<0.01)*1.0)
	
	trig_pred_train_act = numpy.array(numpy.mat(training_set) * numpy.mat(trig_corr))
	trig_pred_validation_act = numpy.array(numpy.mat(validation_set) * numpy.mat(trig_corr))
	
	spont_pred_train_act = numpy.array(numpy.mat(training_set) * numpy.mat(spont_corr))
	spont_pred_validation_act = numpy.array(numpy.mat(validation_set) * numpy.mat(spont_corr))
	
	avg_pred_train_act =  numpy.array(numpy.mat(training_set) * numpy.mat(numpy.ones((z,1))))
	avg_pred_validation_act =  numpy.array(numpy.mat(validation_set) * numpy.mat(numpy.ones((z,1))))
	
	pylab.figure()
	pylab.imshow(trig_corr,interpolation='nearest')
	pylab.colorbar()
	
	
	pylab.figure()
	pylab.imshow(spont_corr,interpolation='nearest')
	pylab.colorbar()
	
	print numpy.shape(trig_pred_train_act)
	print numpy.shape(trig_pred_validation_act)
	print sizex
	print sizey
	print numpy.shape(training_set)
	print numpy.shape(training_inputs)
	
	
	(e,te,c,tc,RFs,pa,pva,corr_coef,corr_coef_tf) = regulerized_inverse_rf(training_inputs,numpy.divide(training_set,trig_pred_train_act),sizex,sizey,__main__.__dict__.get('Alpha',50),numpy.mat(validation_inputs),numpy.divide(validation_set,numpy.mat(spont_pred_validation_act)),contrib.dd.DB(None),True)
	pylab.show()
	
def pearcorr(X):
    X = numpy.array(X)
    import scipy.stats	
    x,y = numpy.shape(X)	
    c = numpy.zeros((y,y))
    p = numpy.zeros((y,y))
    for i in xrange(0,y):
	for j in xrange(0,y):	
	    a,b = scipy.stats.pearsonr(X[:,i],X[:,j])
	    c[i][j]=a
	    p[i][j]=b
	    
    return (c,p)	
	
def rot90_around_center_of_gravity(rf):
    """
    Assumes the RF is in lower right quadrant!!!!!!!!!!!!!!!!
    """	
    sx,sy = numpy.shape(rf)
    cy,cx = centre_of_gravity(rf)
    cx = round(cx)
    cy = round(cy)
    z=numpy.min([cx,cy,sx-cx,sy-cy])
    res = numpy.zeros((sx,sy))
    res[cx-z:cx+z,cy-z:cy+z] = numpy.rot90(rf[cx-z:cx+z,cy-z:cy+z])
    return res


def performance_analysis(training_set,validation_set,pred_act,pred_val_act,raw_validation_set):
    raw_validation_data_set=numpy.rollaxis(numpy.array(raw_validation_set),2)
    
    signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance  = signal_power_test(raw_validation_data_set, numpy.array(training_set), numpy.array(validation_set), numpy.array(pred_act), numpy.array(pred_val_act))
    
    print 'Using all neurons:'
    (ranks,correct,pred) = performIdentification(validation_set,pred_val_act)
    	
    print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power) , " / " , numpy.mean(validation_prediction_power)
    print "Correctly prediced:", correct ,'(', (correct*1.0)/numpy.shape(validation_set)[0]*100 ,'%)' ,  "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set - pred_val_act,2))
		
		
    significant = numpy.mat(numpy.nonzero(((numpy.array(signal_power) - 0.5*numpy.array(signal_power_variance)) > 0.0)*1.0)).getA1() 		
    print 'Using significant neurons:','(', len(significant) ,')'
    (ranks,correct,pred) = performIdentification(validation_set[:,significant],pred_val_act[:,significant])
    
    print "Prediction power on training set / validation set: ", numpy.mean(training_prediction_power[significant]) , " / " , numpy.mean(validation_prediction_power[significant])
    print "Correctly prediced:", correct ,'(', (correct*1.0)/numpy.shape(validation_set)[0]*100 ,'%)' ,  "Mean rank:", numpy.mean(ranks) , "MSE", numpy.mean(numpy.power(validation_set[:,significant] - pred_val_act[:,significant],2))
    
    return (signal_power,noise_power,normalized_noise_power,training_prediction_power,validation_prediction_power,signal_power_variance)
	
