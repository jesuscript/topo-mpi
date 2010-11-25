import contrib.dd
import __main__
import numpy
import pylab

def sortOutLoading(db_node):
    ap = {}
    #ap["dataset"] = 'Gallant_e0008'	
    #ap["dataset"] = '2009_11_04_region3'
    ap["dataset"] = __main__.__dict__.get('dataset','Gallant_e0008')
    db_node = db_node.get_child(ap)
	
    params={}	
    params["clump_mag"] = __main__.__dict__.get('ClumpMag',0.1)
    params["normalize_inputs"] = __main__.__dict__.get('NormalizeInputs',False)
    params["normalize_activities"] = __main__.__dict__.get('NormalizeActivities',False)
    params["cut_out"] = __main__.__dict__.get('CutOut',False)
    params["validation_set_fraction"] = __main__.__dict__.get('ValidationSetFraction',50)		
    params["density"] = __main__.__dict__.get('densit  B y', 0.15)
    params["spiking"] = __main__.__dict__.get('Spiking', True)
    params["2photon"] = __main__.__dict__.get('2photon', True)
    params['LGN'] =  __main__.__dict__.get('LGN', False)
    if params['LGN']:
    	params['LGNCenterSize'] =  __main__.__dict__.get('LGNCenterSize', 2.0) 
    	params['LGNSurrSize'] =  __main__.__dict__.get('LGNSurroundSize', 5.0)
    density= params["density"]
    	
    custom_index=None
    single_file_input = False
	
    if ap["dataset"] == 'Gallant_e0012':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/e0012_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=8000
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "e0012_stim.dat"	
       
    if ap["dataset"] == 'Gallant_e0008':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/e0008_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=8000
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "e0008_stim.dat"
       
    if ap["dataset"] == 'Gallant_r0212b':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r0212b_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=6248
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0212b_stim.dat"	
       
    if ap["dataset"] == 'Gallant_r0212a':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r0212b_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=6269
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0212a_stim.dat"	
   
    if ap["dataset"] == 'Gallant_r0212c':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r0212c_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=6248
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0212c_stim.dat"	
   
    if ap["dataset"] == 'Gallant_r0260':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r0260_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=36000
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0260_stim.dat"	
   
    if ap["dataset"] == 'Gallant_r0279':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r079_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=36000
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0279_stim.dat"	
   
    if ap["dataset"] == 'Gallant_r0284':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r084_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=18000
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0284_stim.dat"	

   
    if ap["dataset"] == 'Gallant_r0301':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r0301_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=27600
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0301_stim.dat"	
    
    if ap["dataset"] == 'Gallant_r0305':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/Gallant/r0305_resp.dat"	
       num_cells = 1
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       inputs_offset=0
       params["2photon"]=False
       num_stim=35996
       #num_stim=100
       single_file_input=True
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Gallant/"
       input_match_string = "r0305_stim.dat"	
    
    
    if ap["dataset"] == '2010_03_12':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/2010_03_12/Exp_nonfilt_dFoF.txt"	
       num_cells = 47    
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1800
       inputs_offset=0
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Stimuli/NG/depackaged/"
       input_match_string = "frame%05d.tif"
		
    if ap["dataset"] == '2010_03_15':
       dataset_loc = "/home/antolikjan/topographica/topographica/Mice/2010_03_15/Exp_nonfilt.txt"	
       num_cells = 51    
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       #num_stim=1708
       num_stim=900
       inputs_offset=0
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Stimuli/NG/depackaged/"
       input_match_string = "frame%05d.tif"
       custom_index = numpy.hstack((numpy.arange(1200,1408,1),numpy.arange(0,300,1),numpy.arange(0,300,1),numpy.arange(600,900,1),numpy.arange(600,900,1),numpy.arange(1200,1500,1)))
		
    if ap["dataset"] == '2010_02_09_noise':
       dataset_loc = "Mice/2010_02_09/2010_02_09_noise_56to58_LowCut_Filtered.txt"	
       num_cells = 53    
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1800
       inputs_offset=0
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Stimuli/SparseNoise_DS=3.0_Step=2_Density=20/depackaged/"
       input_match_string = "frame%05d.tif"
	
    if ap["dataset"] == '2010_02_09_movies':
       dataset_loc = "Mice/2010_02_09/2010_02_09_movies_46to48_LowCut_Filtered.txt"	
       num_cells = 37   
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1800
       inputs_offset=0
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Stimuli/NG/depackaged/"
       input_match_string = "frame%05d.tif"
	
    if ap["dataset"] == '2009_11_04_region3':
       if params["spiking"]:
	  dataset_loc = "Mice/2009_11_04/Raw/region3/spiking_3-7.dat"
	  val_dataset_loc = "Mice/2009_11_04/Raw/region3/val/spiking_3-7.dat"
       else:
	  #dataset_loc = "Mice/2009_11_04/region3_stationary_180_15fr_103cells_on_response_spikes"	
          #val_dataset_loc = "Mice/2009_11_04/region3_50stim_10reps_15fr_103cells_on_response_spikes"
	  dataset_loc = "Mice/2009_11_04/Raw/region3/nospiking_3-7.dat"
	  val_dataset_loc = "Mice/2009_11_04/Raw/region3/val/nospiking_3-7.dat"
       
       #cut_out_x=0.45
       #cut_out_y=0.2
       #cut_out_size=0.7
       cut_out_x=0.3
       cut_out_y=0.0
       cut_out_size=1.0
       
       num_cells = 103    
       sepparate_validation_set = True
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1800
       inputs_offset=1001
       inputs_directory = "/home/antolikjan/topographica/topographica/Flogl/DataOct2009/20090925_image_list_used/"
       input_match_string = "image_%04d.tif"
       val_inputs_directory = "/home/antolikjan/topographica/topographica/Mice/2009_11_04/"
       val_input_match_string = "/20091104_50stimsequence/50stim%04d.tif"
       val_reps = 10

    if ap["dataset"] == '2009_11_04_region5':
       dataset_loc = "Mice/2009_11_04/region5_stationary_180_15fr_103cells_on_response_spikes"	
       val_dataset_loc = "Mice/2009_11_04/region5_50stim_10reps_15fr_103cells_on_response_spikes"
       
       #cut_out_x=0.45
       #cut_out_y=0.2
       #cut_out_size=0.7
       cut_out_x=0.3
       cut_out_y=0.0
       cut_out_size=1.0

       num_cells = 55    
       sepparate_validation_set = True
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1260
       inputs_offset=1001
       inputs_directory = "/home/antolikjan/topographica/topographica/Flogl/DataOct2009/20090925_image_list_used/"
       input_match_string = "image_%04d.tif"
       val_inputs_directory = "/home/antolikjan/topographica/topographica/Mice/2009_11_04/"
       val_input_match_string = "/20091104_50stimsequence/50stim%04d.tif"
       val_reps = 8

    if ap["dataset"] == '20090925_14_36_01':
       dataset_loc = "./Mice/20090925_14_36_01/(20090925_14_36_01)-_retinotopy_region2_sequence_50cells_2700images_spikes"	
       num_cells = 50    
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=2700
       cut_out_x=0.3
       cut_out_y=0.0
       cut_out_size=1.0
       inputs_offset=1001
       inputs_directory = "/home/antolikjan/topographica/topographica/Flogl/DataOct2009/20090925_image_list_used/"
       input_match_string = "image_%04d.tif"

    if ap["dataset"] == '20090924':
       dataset_loc = "Mice/20090924/spiking_3-7.dat"	
       num_cells = 48    
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1620
       cut_out_x=0.2
       cut_out_y=0.0
       cut_out_size=1.0
       inputs_offset=1001
       inputs_directory = "/home/antolikjan/topographica/topographica/Flogl/DataOct2009/20090925_image_list_used/"
       input_match_string = "image_%04d.tif"


    if ap["dataset"] == '20091110_19_16_53':
       dataset_loc = "Mice/20091110_19_16_53/(20091110_19_16_53)-_retinotopy_region4_stationary_180_15fr_66cells_on_response_spikes"	
       num_cells = 68    
       sepparate_validation_set = False
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1440
       cut_out_x=0.05
       cut_out_y=0.0
       cut_out_size=1.0
       inputs_offset=1001
       inputs_directory = "/home/antolikjan/topographica/topographica/Flogl/DataOct2009/20090925_image_list_used/"
       input_match_string = "image_%04d.tif"

    if ap["dataset"] == '2010_04_22':
       if params["spiking"]:
	  dataset_loc = "Mice/2010_04_22/spiking_3-7.dat"	    
       	  val_dataset_loc = "Mice/2010_04_22/val/spiking_3-7.dat"
       else:
          print 'ERROR: no no-spiking data'	    
       num_cells = 102    
       sepparate_validation_set = True
       num_rep=1
       num_frames=1
       transpose=False
       average_frames_from=0
       average_frames_to=1
       num_stim=1800
       inputs_offset=0
       cut_out_x=0.1
       cut_out_y=0.0
       cut_out_size=1.0
       inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Stimuli/NG/1800/depackaged/"
       input_match_string = "frame%05d.tif"
       val_inputs_directory = "/home/antolikjan/topographica/topographica/Mice/Stimuli/NG/1800/depackaged_val/"
       val_input_match_string = "frame%05d.tif"
       val_reps = 12

    db_node = db_node.get_child(params)

    dataset = loadSimpleDataSet(dataset_loc,num_stim,num_cells,num_rep=num_rep,num_frames=num_frames,offset=0,transpose=transpose)
    
    if custom_index != None:
       (index,data) = dataset
       dataset = (custom_index, data)    
    
    
    dataset = averageRangeFrames(dataset,average_frames_from,average_frames_to)
    dataset = averageRepetitions(dataset)
    
    if not sepparate_validation_set:	
	(validation_data_set,dataset) = splitDataset(dataset,params["validation_set_fraction"])
	validation_set = generateTrainingSet(validation_data_set)
	ff = numpy.arange(0,num_cells,1)*0
	if single_file_input:
		validation_inputs=generateInputsFromBinaryFile(validation_data_set,inputs_directory,input_match_string,params["density"])
	else:
		validation_inputs=generateInputs(validation_data_set,inputs_directory,input_match_string,params["density"],offset=inputs_offset)
    else:
	valdataset = loadSimpleDataSet(val_dataset_loc,50,num_cells,val_reps)
	(valdataset,trash) = splitDataset(valdataset,params["validation_set_fraction"])
	flat_valdataset = flattenDataset(valdataset)
	ff  = analyse_reliability(valdataset,params)
	#get rid of frames and make it into a 3D stack where first dimension is repetitions
	from copy import deepcopy
	(index,raw_val_set) = valdataset
	rr=[]
	for i in xrange(0,val_reps):
	    rr.append(generateTrainingSet(averageRepetitions((index,deepcopy(raw_val_set)),reps=[i])))
	raw_val_set = rr
	if params["spiking"] and params["2photon"]:
	   for i in xrange(0,val_reps):
	       raw_val_set[i] = (raw_val_set[i])/0.028	    
	valdataset = averageRangeFrames(valdataset,0,1)
    	valdataset = averageRepetitions(valdataset)
        validation_set = generateTrainingSet(valdataset)
	if single_file_input:
		validation_inputs=generateInputsFromBinaryFil(valdataset,val_inputs_directory,val_input_match_string,params["density"])
	else:    		
		validation_inputs=generateInputs(valdataset,val_inputs_directory,val_input_match_string,params["density"],offset=0)
	flat_validation_set = generateTrainingSet(flat_valdataset)
    	flat_validation_inputs=generateInputs(flat_valdataset,val_inputs_directory,val_input_match_string,params["density"],offset=0)
	
    

    training_set = generateTrainingSet(dataset)
    if single_file_input:
	training_inputs=generateInputsFromBinaryFile(dataset,inputs_directory,input_match_string,params["density"])
    else:
    	training_inputs=generateInputs(dataset,inputs_directory,input_match_string,params["density"],offset=inputs_offset)
    
    if params["spiking"] and params["2photon"]:
    	training_set = (training_set)/0.028
    	validation_set = (validation_set)/0.028
    
    if params["normalize_inputs"]:
       training_inputs = ((numpy.array(training_inputs) - 128.0)/128.0) * numpy.sqrt(2) 
       validation_inputs = ((numpy.array(validation_inputs) - 128.0)/128.0) * numpy.sqrt(2)
    
    training_inputs=numpy.array(training_inputs)/1000000.0
    validation_inputs=numpy.array(validation_inputs)/1000000.0
    
    
    if params["normalize_activities"]:
        (a,v) = compute_average_min_max(training_set)
        training_set = normalize_data_set(training_set,a,v)
        validation_set = normalize_data_set(validation_set,a,v)
	if sepparate_validation_set:
		for i in xrange(0,val_reps):
		      raw_val_set[i] = normalize_data_set(raw_val_set[i],a,v)
    
    if params["cut_out"]:
        (x,y)= numpy.shape(training_inputs[0])
        training_inputs = cut_out_images_set(training_inputs,int(x*cut_out_size),(int(x*cut_out_y),int(y*cut_out_x)))
        validation_inputs = cut_out_images_set(validation_inputs,int(x*cut_out_size),(int(x*cut_out_y),int(y*cut_out_x)))
    
    (sizex,sizey) = numpy.shape(training_inputs[0])
    print (sizex,sizey)
    
    if __main__.__dict__.get('LGN', False):
	training_inputs = LGN_preprocessing(training_inputs,params['LGNCenterSize'],params['LGNSurrSize'])    
	validation_inputs = LGN_preprocessing(validation_inputs,params['LGNCenterSize'],params['LGNSurrSize'])
    
    training_inputs = generate_raw_training_set(training_inputs)
    validation_inputs = generate_raw_training_set(validation_inputs)
    
    db_node.add_data("training_inputs",training_inputs,force=True)
    db_node.add_data("training_set",training_set,force=True)
    db_node.add_data("validation_inputs",validation_inputs,force=True)
    db_node.add_data("validation_set",validation_set,force=True)
    db_node.add_data("Fano Factors",ff,force=True)
    if sepparate_validation_set:
    	db_node.add_data("raw_validation_set",raw_val_set,force=True)
	db_node.add_data("flat_validation_inputs",flat_validation_inputs,force=True)
	db_node.add_data("flat_validation_set",flat_validation_set,force=True)
    else:
	db_node.add_data("raw_validation_set",[validation_set],force=True)	    
    #pylab.figure()
    #pylab.plot(training_set,'o')
    
    print "Training set size:"
    print numpy.shape(training_set)
    return (sizex,sizey,training_inputs,training_set,validation_inputs,validation_set,ff,db_node)
    
    
    
    
    
    
    
    
    
def loadSimpleDataSet(filename,num_stimuli,n_cells,num_rep=1,num_frames=1,offset=0,transpose=False):
    f = file(filename, "r") 
    data = [line.split() for line in f]
    if transpose:
       data = numpy.array(data).transpose()	    
    
    f.close()
    print "Dataset shape:", numpy.shape(data)

    dataset = [([[] for i in xrange(0,num_stimuli)]) for j in xrange(0,n_cells)]
    
    for k in xrange(0,n_cells):
        for i in xrange(0,num_stimuli):
	    for rep in xrange(0,num_rep):
		f = []
            	for fr in xrange(0,num_frames):
                       f.append(float(data[rep*num_stimuli+i*num_frames+fr+offset][k]))
            	dataset[k][i].append(f)
    print numpy.shape(dataset)
    return (numpy.arange(0,num_stimuli),dataset)
            

def loadRandomizedDataSet(directory,num_rep,num_frames,num_stimuli,n_cells):
    f = file(directory + "/combined_data", "r") 
    data = [line.split() for line in f]
    f.close()

    f = file(directory +"/image_sequence", "r") 
    random = [line.split() for line in f]
    random=numpy.array(random)
    random = random[0]
    f.close()
    r=[]
    for j in random:
        r.append(int(float(j)))
    random = r

    dataset = [([[] for i in xrange(0,num_stimuli)]) for i in xrange(0,n_cells)]

    (recordings,num_cells) = numpy.shape(data)

    assert recordings == (num_rep * num_stimuli * num_frames)
    assert recordings / num_frames == len(random)
    assert n_cells == num_cells
    
    for k in xrange(0,num_cells):
        for i in xrange(0,num_rep*num_stimuli):
            f = []
            for fr in xrange(0,num_frames):
                        f.append(float(data[i*num_frames+fr][k]))
            dataset[k][random[i]-1].append(f)

    return (numpy.arange(0,num_stimuli),dataset)

def mergeDatasets(dataset1,dataset2):
    print "Warning: Indexes must match"
    (index,data1) = dataset1
    (index,data2) = dataset2
    data1.extend(data2)
    return (index,data1)
 
def averageRangeFrames(dataset,min,max):
    (index,data) = dataset
    
    for cell in data:
        for stimulus in cell:
            for r in xrange(0,len(stimulus)):
                stimulus[r]=[numpy.average(stimulus[r][min:max])]
    
    return (index,data) 		

def averageRepetitions(dataset,reps=None):
    (index,data) = dataset
    (num_cells,num_stim,num_rep,num_frames) = numpy.shape(data)
    if reps==None:
       reps = numpy.arange(0,num_rep,1)

    for cell in data:
        for stimulus in xrange(0,num_stim):
            r = [0 for i in range(0,num_frames)]
            for rep in reps:
                for f in xrange(0,num_frames):
                    r[f]+=cell[stimulus][rep][f]/(len(reps)*1.0)
	    
            cell[stimulus]=[r]
    return (index,data)

def analyse_reliability(dataset,params):
    (index,data) = dataset
    c = []

    for cell in data:
	fano_factors=[]    
	for stimuli in cell:
	    z = [] 	
	    for rep in stimuli:
		z.append(numpy.mean(rep))
	    z=numpy.array(z)
	    fano_factors.append(numpy.array(z).var()/numpy.mean(z))
	c.append(numpy.mean(fano_factors))
    
    #pylab.figure()
    #pylab.hist(c)
    return c

def splitDataset(dataset,ratio):
    (index,data) = dataset
    (num_cells,num_stim,trash1,trash2) = numpy.shape(data)

    dataset1=[]
    dataset2=[]
    index1=[]
    index2=[]

    if ratio<=1.0:
    	tresh = num_stim*ratio
    else:
	tresh = ratio

    for i in xrange(0,num_stim):
        if i < numpy.floor(tresh):
            index1.append(index[i])
        else:    
            index2.append(index[i])
    
    for cell in data:
        d1=[]
        d2=[]
        for i in xrange(0,num_stim):
            if i < numpy.floor(tresh):
               d1.append(cell[i])
            else:    
               d2.append(cell[i])
        dataset1.append(d1)
        dataset2.append(d2)

    return ((index1,dataset1),(index2,dataset2))

def generateTrainingSet(dataset):
    (index,data) = dataset

    training_set=[]
    for cell in data:
        cell_set=[]
        for stimuli in cell:
           for rep in stimuli:
                for frame in rep:
                    cell_set.append(frame)
        training_set.append(cell_set)
    return numpy.array(numpy.matrix(training_set).T)

def flattenDataset(dataset):
    (index,data) = dataset
    (num_cells,num_stim,num_rep,num_frames) = numpy.shape(data)
    data_new = numpy.zeros((num_cells,num_stim*num_rep*num_frames,1,1))
    index_new = []

    z=0
    for j in xrange(0,num_stim):
	for k in xrange(0,num_rep):		
	    for l in xrange(0,num_frames):
		index_new.append(index[j])
		for i in xrange(0,num_cells):			
    		    data_new[i][z][0][0]=data[i][j][k][l]
		z=z+1
    return (index_new,data_new)

def generateInputs(dataset,directory,image_matching_string,density,offset):
    (index,data) = dataset
    import PIL
    import Image
    # ALERT ALERT ALERT We do not handle repetitions yet!!!!!
    image_filenames=[directory+image_matching_string %(i+offset) for i in index]
    ins = []
    for j in xrange(0,len(index)):
        #inputs[j].pattern_sampler.whole_pattern_output_fns=[]
	image = Image.open(image_filenames[j])
	(width,height) = image.size
        inp = image.resize((int(width*density), int(height*density)), Image.ANTIALIAS)
	ins.append(numpy.array(inp.getdata()).reshape( int(height*density),int(width*density)))
 
    return ins


def generateInputsFromBinaryFile(dataset,directory,image_matching_string,density):
    (index,data) = dataset	
    
    f = file(directory + image_matching_string, "r") 
    data = [numpy.array(line.split()) for line in f]
    f.close()

    ins = []
    for j in index:
	b = data[j]
	z = []
	for i in xrange(0,len(b)):
	    z.append(float(b[i]))
        s=numpy.sqrt(len(b))	    
        ins.append(numpy.reshape(numpy.array(z),(s,s))) 
    return ins



def cut_out_images_set(inputs,size,pos):
    (sizex,sizey) = numpy.shape(inputs[0])

    print sizex,sizey
    print size,pos
    (x,y) = pos
    inp = []
    if (x+size <= sizex) and (y+size <= sizey):
        for i in inputs:
                inp.append(i[x:x+size,y:y+size])
    else:
        print "cut_out_images_set: out of bounds"
    return inp

def generate_raw_training_set(inputs):
    out = []
    for i in inputs:
        out.append(i.flatten())
    return numpy.array(out)

def compute_average_min_max(data_set):
    avg = numpy.zeros(numpy.shape(data_set[0]))
    var = numpy.zeros(numpy.shape(data_set[0]))
    
    for d in data_set:
        avg += d
    avg = avg/(len(data_set)*1.0)
    
    for d in data_set:
        var += numpy.multiply((d-avg),(d-avg))
    var = var/(len(data_set)*1.0)
    return (avg,var)
    
def normalize_data_set(data_set,avg,var):
    print numpy.shape(avg)
    for i in xrange(0,len(data_set)):
        data_set[i]-=avg
        data_set[i]=numpy.divide(data_set[i],numpy.sqrt(var)) 
    return data_set

def compute_average_input(inputs):
    avgRF = numpy.zeros(numpy.shape(inputs[0]))

    for i in inputs:
        avgRF += i
    avgRF = avgRF/(len(inputs)*1.0)
    return avgRF

def normalize_image_inputs(inputs,avgRF):
    for i in xrange(0,len(inputs)):
        inputs[i]=inputs[i]-avgRF

    return inputs


def LGN_preprocessing(images,center_size,surr_size):
    import scipy.signal
    sizex,sizey= numpy.shape(images[0])	 
    new_images = numpy.zeros(numpy.shape(images))	
	
    xx = numpy.repeat([numpy.arange(0,sizex,1)],sizex,axis=0).T	
    yy = numpy.repeat([numpy.arange(0,sizex,1)],sizex,axis=0)
    kernel = numpy.exp(-((xx - sizex/2)**2 + (yy - sizex/2)**2)/center_size)/(numpy.pi*center_size) - numpy.exp(-((xx - sizex/2)**2 + (yy - sizex/2)**2)/surr_size)/(numpy.pi*surr_size)
    
    for i in xrange(0,len(images)):
	new_images[i,:,:]  = scipy.signal.convolve2d(images[i],kernel,mode='same')
    
    return new_images
	