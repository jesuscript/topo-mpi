from mpi4py import MPI

import numpy as np,copy


from topo.base.cf import CFProjection, MaskedCFIter

from topo import transferfn
import topo.transferfn.projfn



activity_sum = None



class MPI_CFProjection_node(CFProjection):
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.input_buffer = None

        #self.activity_copy = numpy.array([])
        
        

    #KKALERT: same as CFProjection activate except for return
    def activate(self,input_activity):
        """Activate using the specified response_fn and output_fn."""

        self.input_buffer = input_activity
        self.activity *=0.0

        self.response_fn(MaskedCFIter(self), input_activity, self.activity, self.strength)

        for of in self.output_fns:
            of(self.activity)

        return [self.activity,MPI.DOUBLE]
            

    def learn(self):
        if self.input_buffer != None:
            dest_activity = self.get_dest_activity_opt()
            
            self.learning_fn(MaskedCFIter(self),self.input_buffer,dest_activity,self.learning_rate)
        

    # KKALERT: overridden in order to maintain consistency with the original call format
    # i.e. to pass active_units_mask as a _named_ parameter
    def apply_learn_output_fns(self, active_units_mask):
        super(MPI_CFProjection_node,self).apply_learn_output_fns(active_units_mask=active_units_mask)


    
    def add_to_activity_sum(self):
        if activity_sum==None:
            activity_sum=self.activity

    """>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> GETTERS AND SETTERS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    def get_dest_mask(self):
        return self.mask
    
    
    def set_dest(self,dest):
        self.dest_ref = dest

    def get_dest_activity_opt(self):
        self.comm = MPI.COMM_WORLD

        if self.rank == 0:
            activity = self.dest_ref.activity
            items_per_node = int(round(activity.size/self.size))
            data = []
            start_i = 0
            for i in range(self.size):
                if i+1 < self.size:
                    data.append(activity.flat[start_i : start_i+items_per_node])
                else:
                    data.append(activity.flat[start_i : ])
                start_i+=items_per_node
        else:
            data = None
        
        activity = self.comm.scatter(data, root = 0)
        return activity
        

    def get_dest_activity(self):
        if self.rank == 0:
            activity = self.dest_ref.activity
        else:
            activity = None

        activity = self.comm.bcast(obj=activity, root=0)
        
        items_per_node = int(round(activity.size/self.size))
        if self.rank+1<self.size:
            activity_chunk = activity.reshape(1,activity.size)[0][self.rank * items_per_node : (self.rank+1) * items_per_node]
        else:
            activity_chunk = activity.reshape(1,activity.size)[0][self.rank * items_per_node : ]
        return activity_chunk
    
    
    def _set_flatcfs_ref(self, flatcfs_ref):
        self.flatcfs_ref = flatcfs_ref
    def _set_flatcfs_chunk(self):
        if self.rank == 0:
            #there should be a better way of distributing these
            items_per_node = len(self.flatcfs_ref)/self.size
            data = []
            start_i = 0
            for i in range(self.size):
                if i+1 < self.size:
                    data.append(self.flatcfs_ref[start_i : start_i+items_per_node])
                else:
                    data.append(self.flatcfs_ref[start_i : ])
                start_i+=items_per_node
        else:
            data = None
        self.flatcfs = self.comm.scatter(data, root = 0)
    def _get_flatcfs_chunk(self):
        return self.flatcfs

    
    def _set_activity(self, activity):
        if activity==None:
            self.activity = None
        else:
            activity_copy = activity.copy()
            activity_copy = activity_copy.ravel()
            items_per_node = activity_copy.size / self.size
            if self.rank+1<self.size:
                self.activity = activity_copy[self.rank * items_per_node : (self.rank+1) * items_per_node]
            else:
                self.activity = activity_copy[self.rank * items_per_node : ]

            if self.rank==0:
                #recvcounts and displacements, for MPI Gatherv
                counts = np.array([items_per_node]*(self.size-1) + 
                                  [activity_copy.size - items_per_node*(self.size-1)])
                displs = np.arange(0,activity_copy.size,items_per_node)[:self.size]
                return [activity_copy,(counts,displs),MPI.DOUBLE]
            #self.activity_copy = self.activity.copy()
            

    

    def _set_dest_mask(self,dest_mask):
        if dest_mask == None:
            self.mask = None
        else:
            items_per_node = int(round(dest_mask.data.size / self.size))
            self.mask = copy.copy(dest_mask)
            self.mask.data = self.mask.data.reshape(1,self.mask.data.size)
            if self.rank+1<self.size:
                self.mask.data = self.mask.data[0][self.rank * items_per_node : (self.rank+1) * items_per_node]
            else:
                self.mask.data = self.mask.data[0][self.rank * items_per_node :]
    def _get_dest_mask(self):
        return self.mask
        
    def _set_n_units(self,n_units):
        self.n_units = n_units
    def _get_n_units(self):
        return self.n_units
    
    def _set_strength(self,strength):
        self.strength = strength
    def _get_strength(self):
        return self.strength
    
    def _set_allow_skip_non_responding_units(self,allow):
        self.allow_skip_non_responding_units = allow
    def _get_allow_skip_non_responding_units(self):
        return self.allow_skip_non_responding_units
    
    def set_sub_activate_time(self, activate_time):
        self.sub_activate_time = activate_time
    def get_sub_activate_time(self):
        return self.sub_activate_time
    
    def set_response_fn(self, response_fn):
        self.response_fn = response_fn
    def get_response_fn(self):
        return self.response_fn
    
    def set_weights_output_fns(self,weights_output_fns):
        self.weights_output_fns = weights_output_fns
    def get_weights_output_fns(self):
        return self.weights_output_fns
    
    def set_learning_rate(self,learning_rate):
        self.learning_rate = learning_rate
    def get_learning_rate(self):
        return self.learning_rate
    
    def set_nominal_bounds_template(self,nominal_bounds_template):
        self.nominal_bounds_template = nominal_bounds_template
    def get_nominal_bounds_template(self):
        return self.nominal_bounds_template
    
    def set_learning_fn(self,learning_fn):
        self.learning_fn = learning_fn
    def get_learning_fn(self):
        return self.learning_fn
    
    def set_cf_shape(self,cf_shape):
        self.cf_shape = cf_shape
    def get_cf_shape(self):
        return self.cf_shape
    
    def set_mask_threshold(self,mask_threshold):
        self.mask_threshold = mask_threshold
    def get_mask_threshold(self):
        return self.mask_threshold
    
    def set_coord_mapper(self, coord_mapper):
        self.coord_mapper = coord_mapper
    def get_coord_mapper(self):
        return self.coord_mapper
    
    def set_scsfac(self, scsfac):
        self.same_cf_shape_for_all_cfs = scsfac
    def get_scsfac(self):
        return self.same_cf_shape_for_all_cfs
    
    def set_allow_null_cfs(self, allow_null_cfs):
        self.allow_null_cfs = allow_null_cfs
    def get_allow_null_cfs(self):
        return self.allow_null_cfs
    
    def set_autosize_mask(self,autosize_mask):
        self.autosize_mask = autosize_mask
    def get_autosize_mask(self):
        return self.autosize_mask
    
    def set_apply_output_fns_init(self,apply_output_fns_init):
        self.apply_output_fns_init = apply_output_fns_init
    def get_apply_output_fns_init(self):
        return self.apply_output_fns_init
    
    def set_min_matrix_radius(self,min_matrix_radius):
        self.min_matrix_radius = min_matrix_radius
    def get_min_matrix_radius(self):
        return self.min_matrix_radius
    
    
    
    def set_masked_norm_totals(self,norm_totals):
        if self.rank == 0:
            data = self.norm_sizes
        else:
            data = None
        (start_i, end_i) = self.comm.scatter(data, root=0)
        norm_totals_chunk = norm_totals[start_i:end_i]
        super(MPI_CFProjection_node,self).set_masked_norm_totals(norm_totals_chunk)
            
    def get_masked_norm_totals(self,active_units_mask):
        norm_totals = super(MPI_CFProjection_node,self).get_masked_norm_totals(active_units_mask=active_units_mask)
        norm_size = len(norm_totals)
        
        # the following algorithm saves the number of unmasked non-totals on each node in order
        # to distribute them on set_masked_norm_totals the same way they were gathered
        norm_size = self.comm.gather(norm_size, root=0)
        
        if self.rank == 0:
            self.norm_sizes = []
            start_i = 0
            for n in norm_size:
                self.norm_sizes.append((start_i, start_i+n))
                start_i += n
        
        return norm_totals

    def set_fake_src(self,fake_src):
        self.fake_src = fake_src
    def get_fake_src(self):
        return self.fake_src

    def set_mask_template(self,mask_template):
        self.mask_template = mask_template
    def get_mask_template(self):
        return self.mask_template

    def set__slice_template(self,_slice_template):
        self._slice_template = _slice_template
    def get__slice_template(self):
        return self._slice_template

    """<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< GETTERS AND SETTERS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"""
    
