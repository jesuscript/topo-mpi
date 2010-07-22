# Adding support for RGB images (work in progress)
import topo
import param

# CB: it's like a NumberGenerator, but it isn't one, so
# I'm not sure where it should go.
class ListGenerator(param.Parameterized):
    """
    When called, returns a list of the results of calling all its
    generators.
    """
    generators = param.List(default=[],doc="""
        List of callables used to produce the ListGenerator's items.""")

    def __call__(self):
        return [g() for g in self.generators]



from topo.sheet.basic import GeneratorSheet,FunctionEvent,PeriodicEventSequence,param,PatternGenerator
from topo import pattern

class ColorImageSheet(GeneratorSheet):
    """
    A GeneratorSheet that handles RGB images.

    Accepts either a single-channel or an RGB input_generator.  If the
    input_generator stores separate red, green, and blue patterns, it
    is used as-is; other (monochrome) PatternGenerators are first
    wrapped using ExtendToRGB to create the RGB patterns.

    When a pattern is generated, a monochrome version is sent out on
    the Activity port as usual for a GeneratorSheet, and red, green,
    and blue activities are sent out on the RedActivity,
    GreenActivity, and BlueActivity ports.  Thus this class can be used
    just like GeneratorSheet, but with optional color channels.
    """

    src_ports=['Activity','RedActivity','GreenActivity','BlueActivity']

    def __init__(self,**params):
        super(ColorImageSheet,self).__init__(**params)
        self.activity_red=self.activity.copy()
        self.activity_green=self.activity.copy()
        self.activity_blue=self.activity.copy()


    def set_input_generator(self,new_ig,push_existing=False):
        """Wrap new_ig in ExtendToRGB if necessary."""
        # CEBALERT: this conditional wrapping of the input generator
        # is confusing.  Why have logic for supporting non-RGB
        # patterns in ExtendToRGB and in ColorImageSheet? (Comes down
        # to problem of accessing parameters of subgenerators.)
        if not hasattr(new_ig,'red'):
            new_ig = ExtendToRGB(generator=new_ig)
            
        super(ColorImageSheet,self).set_input_generator(new_ig,push_existing=push_existing)

        
    def generate(self):
        """
        Works as in the superclass, but also generates RGB output and sends
        it out on the RedActivity, GreenActivity, and BlueActivity ports.
        """
        super(ColorImageSheet,self).generate()
        
        self.activity_red[:]   = self.input_generator.red
        self.activity_green[:] = self.input_generator.green
        self.activity_blue[:]  = self.input_generator.blue
        
        if self.apply_output_fns:
            for output_fn in self.output_fns:
                output_fn(self.activity_red)
                output_fn(self.activity_green)
                output_fn(self.activity_blue)

        self.send_output(src_port='RedActivity',  data=self.activity_red)
        self.send_output(src_port='GreenActivity',data=self.activity_green)
        self.send_output(src_port='BlueActivity', data=self.activity_blue)




import copy
from param.parameterized import ParamOverrides
class ExtendToRGB(PatternGenerator):
    """
    Wrapper for any PatternGenerator to support red, green, and blue
    channels, e.g. for use with ColorImageSheet.

    If the specified generator itself has a 'generator' attribute,
    ExtendToRGB will attempt to get red, green, and blue from
    generator.generator (e.g. ColorImage inside a Selector);
    otherwise, ExtendToRGB will attempt to get red, green, and blue
    from generator. If no red, green, and blue are found in these
    ways, ExtendToRGB will synthesize the red, green, and blue
    channels.

    After finding or synthesizing red, green, and blue, they are
    scaled according to relative_channel_strengths.
    """
    channels = ["red","green","blue"]

    generator = param.ClassSelector(class_=pattern.PatternGenerator,
                                    default=pattern.Constant())

    # CEB: not a tuple, because tuple elements cannot be changed
    relative_channel_strengths = param.Dynamic(default=[1.0,1.0,1.0],doc="""
        Scaling of each channel relative to the others.""") 

    def __init__(self,**params):
        super(ExtendToRGB,self).__init__(**params)
        for c in self.channels:
            setattr(self,c,None)

    def __call__(self,**params):
        p = ParamOverrides(self,params)

        # as for Selector etc, pass through certain parameters to
        # generator
        params['xdensity']=p.xdensity
        params['ydensity']=p.ydensity
        params['bounds']=p.bounds

        # (not **p because that would be extra parameters)
        gray = p.generator(**params)


        # CEB: method more complicated than it needs to be; maybe if
        # the various selector pattern generators had a way of
        # accessing the current generator's parameters, it could be
        # simpler?

        # got to get the generator that's actually making the pattern!
        if hasattr(p.generator,'get_current_generator'):
            # access the generator without causing any index to be advanced
            generator = p.generator.get_current_generator()
        elif hasattr(p.generator,'generator'):
            # CB: could at least add appropriate
            # get_current_generator() to patterns other than Selector,
            # like Translator etc
            generator = p.generator.generator
        else:
            generator = p.generator

        # Promote red, green, blue from 'actual generator' if it
        # has them. Otherwise set red, green, blue to gray/3.
        n_channels = len(self.channels)
        channel_values=([gray]*n_channels if not hasattr(generator,'red') else \
                        [getattr(generator,channel) for channel in self.channels])

        total_strength = sum(p.relative_channel_strengths)
        if total_strength==0:
            total_strength=1.0

        for name,value,strength in zip(p.channels,channel_values,p.relative_channel_strengths):
            setattr(self,name,n_channels*value*strength/total_strength)

        return gray

        
from topo.pattern.image import FileImage,edge_average,PIL
import ImageOps, ImageEnhance
import numpy

from contrib.rgbhsv import rgb_to_hsv_array_opt as rgb_to_hsv_array, \
                           hsv_to_rgb_array_opt as hsv_to_rgb_array

from topo import transferfn

# CEB: might be simpler to use HSV internally and provide red, green,
# and blue properties (that do and cache a conversion)...but PIL
# doesn't support HSV.

class ColorImage(FileImage):
    """
    A FileImage that handles RGB color images.
    """
    def _get_image(self,p):
        if p.filename!=self.last_filename or self._image is None:
            self.last_filename=p.filename
            rgbimage = PIL.open(p.filename)
            try:
                R,G,B = rgbimage.split()
            except ValueError:
                # grayscale image, so just put 1/3 in each channel
                R=G=B=ImageEnhance.Brightness(ImageOps.grayscale(rgbimage)).enhance(1.0/3.0)
            self._image_red  = R
            self._image_green = G
            self._image_blue = B
            self._image = ImageOps.grayscale(rgbimage)
        return self._image


    def __call__(self,**params_to_override):
        # Uses super's call for grayscale and to set up self.red,
        # self.green, self.blue, then performs the same actions
        # as the super's call on the separate color channels.

        # PatternGenerator.__call__ does:
        # function, apply_mask, scale+offset, outputfns
        gray = super(ColorImage,self).__call__(**params_to_override)
        p = ParamOverrides(self,params_to_override)
        for M in (self.red,self.green,self.blue):
            self._apply_mask(p,M)
            M*=p.scale
            M+=p.offset

        # any output_fn would need to be applied to the V channel
        # only, so for now just
        assert len(p.output_fns)==0

        if p.cache_image is False:
            self._image_red=self._image_green=self._image_blue=None

        return gray


    def function(self,p):
        """
        In addition to returning grayscale, stores red, green, and
        blue components.
        """
        gray = super(ColorImage,self).function(p)

        orig_image = self._image
        # now store red, green, blue
        # (by repeating the super's function call, but each time first
        # setting _image to the appropriate channel's image)
        self._image = self._image_red
        self.red = super(ColorImage,self).function(p)

        self._image = self._image_green
        self.green = super(ColorImage,self).function(p)

        self._image = self._image_blue
        self.blue = super(ColorImage,self).function(p)

        self._image = orig_image
        # CEBALERT: currently, red, green, blue arrays are cached
        return gray

    
from topo import numbergen
class RotatedHuesImage(ColorImage):
    """
    A ColorImage that rotates the hues in the image by a random value.
    """

    random_generator = param.Callable(
        default=numbergen.UniformRandom(lbound=0,ubound=1))
    
    def function(self,p):
        gray = super(RotatedHuesImage,self).function(p)

        H,S,V = rgb_to_hsv_array(self.red,self.green,self.blue)

        H+=self.random_generator()
        H%=1.0

        self.red,self.green,self.blue = hsv_to_rgb_array(H,S,V)

        return gray



from contrib.cbmisc import OnlineAnalyser, Histogrammer, Summer

def hue_from_rgb(**data):
    """
    Helper function: given arrays of RedActivity, GreenActivity, and
    BlueActivity (values in [0,1]), return an array of the
    corresponding hues.
    """
    red,green,blue = data['RedActivity'],data['GreenActivity'],data['BlueActivity']
    return rgb_to_hsv_array(red,green,blue)[0]


    

if __name__=="__main__" or __name__=="__mynamespace__":

    from topo import sheet
    import glob
    image_filenames = glob.glob('/disk/scratch/fast/v1cball/mcgill/foilage/*.tif') # sic
    images0 = [ColorImage(filename=f) for f in image_filenames]
    images1 = [RotatedHuesImage(filename=f) for f in image_filenames]
    
    input_generator0 = pattern.Selector(generators=images0)
    input_generator1 = pattern.Selector(generators=images1)

    topo.sim['Retina0']=ColorImageSheet(input_generator=input_generator0,nominal_density=48)
    topo.sim['Retina1']=ColorImageSheet(input_generator=input_generator1,nominal_density=48)

    cone_types = ['Red','Green','Blue']
    for c in cone_types:
        for i in range(0,2):
            topo.sim[c+str(i)]=sheet.ActivityCopy(nominal_density=48)
            topo.sim.connect('Retina'+str(i),c+str(i),src_port='%sActivity'%c,dest_port='Activity')


    ## examples of online analysis
    topo.sim['A'] = OnlineAnalyser(
        data_analysis_fn=Histogrammer(data_transform_fn=hue_from_rgb),
        operator_=numpy.add,
        dest_ports=['RedActivity','GreenActivity','BlueActivity'])

    for c in cone_types:
        topo.sim.connect('Retina1','A',name='h%s'%c,
                         src_port='%sActivity'%c,
                         dest_port='%sActivity'%c)


    topo.sim['B'] = OnlineAnalyser(
        data_analysis_fn=Histogrammer(),
        operator_=numpy.add)

    topo.sim.connect('Retina1','B',src_port='Activity')


    topo.sim['C'] = OnlineAnalyser(
        data_analysis_fn=Summer(),
        operator_=numpy.add)

    topo.sim.connect('Retina1','C',src_port='Activity')
