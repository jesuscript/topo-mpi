"""
Array-compatible and optimized versions of the colorsys module's
rgb_to_hsv() and hsv_to_rgb() functions.

$Id$
"""

from topo.misc.inlinec import inline
import numpy


def _check_dims(A,B,C):
    shape = A.shape
    for M in (A,B,C):
        assert M.shape==shape
        assert M.min()>=0.0 and M.max()<=1.0, "Values must be on [0,1] (actual min,max=%s,%s)"%(M.min(),M.max())
    return shape


def rgb_to_hsv_array(r,g,b):
    """
    Equivalent to colorsys.rgb_to_hsv, except:
      * acts on arrays of red, green, and blue pixels
      * asserts (rather than assumes) that inputs are on [0,1]
      * returns arrays of type numpy.float32 for hue, saturation, and
        value
    """
    shape = _check_dims(r,g,b)
    
    from colorsys import rgb_to_hsv
    h=numpy.zeros(shape,dtype=numpy.float32)
    v=numpy.zeros(shape,dtype=numpy.float32)
    s=numpy.zeros(shape,dtype=numpy.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            h[i,j],s[i,j],v[i,j]=rgb_to_hsv(r[i,j],g[i,j],b[i,j])
    return h,s,v


def hsv_to_rgb_array(h,s,v):
    """
    Equivalent to colorsys.hsv_to_rgb, except:
      * acts on arrays of hue, saturation, and value
      * asserts (rather than assumes) that inputs are on [0,1]
      * returns arrays of type numpy.float32 for red, green, and blue
    """
    shape = _check_dims(h,s,v)
    
    from colorsys import hsv_to_rgb
    r = numpy.zeros(shape,dtype=numpy.float32)
    g = numpy.zeros(shape,dtype=numpy.float32)
    b = numpy.zeros(shape,dtype=numpy.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            r[i,j],g[i,j],b[i,j]=hsv_to_rgb(h[i,j],s[i,j],v[i,j])
    return r,g,b



def rgb_to_hsv_array_opt(red,grn,blu):
    """Supposed to be equivalent to rgb_to_hsv_array()."""
    shape = _check_dims(red,grn,blu)
    
    hue = numpy.zeros(shape,dtype=numpy.float32)
    sat = numpy.zeros(shape,dtype=numpy.float32)
    val = numpy.zeros(shape,dtype=numpy.float32)

    code = """
//// MIN3,MAX3 macros from
// http://en.literateprograms.org/RGB_to_HSV_color_space_conversion_(C)
#define MIN3(x,y,z)  ((y) <= (z) ? \
                         ((x) <= (y) ? (x) : (y)) \
                     : \
                         ((x) <= (z) ? (x) : (z)))

#define MAX3(x,y,z)  ((y) >= (z) ? \
                         ((x) >= (y) ? (x) : (y)) \
                     : \
                         ((x) >= (z) ? (x) : (z)))
////

for (int i=0; i<Nred[0]; ++i) {
    for (int j=0; j<Nred[1]; ++j) {

        // translation of Python's colorsys.rgb_to_hsv()

        float r=RED2(i,j);
        float g=GRN2(i,j);
        float b=BLU2(i,j);

        float minc=MIN3(r,g,b); 
        float maxc=MAX3(r,g,b); 

        VAL2(i,j)=maxc;

        if(minc==maxc) {
            HUE2(i,j)=0.0;
            SAT2(i,j)=0.0;
        } else {
            float delta=maxc-minc; 
            SAT2(i,j)=delta/maxc;

            float rc=(maxc-r)/delta;
            float gc=(maxc-g)/delta;
            float bc=(maxc-b)/delta;

            if(r==maxc)
                HUE2(i,j)=bc-gc;
            else if(g==maxc)
                HUE2(i,j)=2.0+rc-bc;
            else
                HUE2(i,j)=4.0+gc-rc;

            HUE2(i,j)=(HUE2(i,j)/6.0);

            if(HUE2(i,j)<0)
                HUE2(i,j)+=1;
            //else if(HUE2(i,j)>1)
            //    HUE2(i,j)-=1;

        }

    }
}

"""
    inline(code, ['red','grn','blu','hue','sat','val'], local_dict=locals())
    return hue,sat,val



def hsv_to_rgb_array_opt(hue,sat,val):
    """Supposed to be equivalent to hsv_to_rgb_array()."""
    shape = _check_dims(hue,sat,val)
    
    red = numpy.zeros(shape,dtype=numpy.float32)
    grn = numpy.zeros(shape,dtype=numpy.float32)
    blu = numpy.zeros(shape,dtype=numpy.float32)

    code = """
for (int i=0; i<Nhue[0]; ++i) {
    for (int j=0; j<Nhue[1]; ++j) {

        // translation of Python's colorsys.hsv_to_rgb() using parts
        // of code from
        // http://www.cs.rit.edu/~ncs/color/t_convert.html
        float h=HUE2(i,j);
        float s=SAT2(i,j);
        float v=VAL2(i,j);

        float r,g,b;
        
        if(s==0) 
            r=g=b=v;
        else {
            int i=(int)floor(h*6.0);
            if(i<0) i=0;
            
            float f=(h*6.0)-i;
            float p=v*(1.0-s);
            float q=v*(1.0-s*f);
            float t=v*(1.0-s*(1-f));

            switch(i) {
                case 0:
                    r = v;
                    g = t;
                    b = p;
                    break;
                case 1:
                    r = q;
                    g = v;
                    b = p;
                    break;
                case 2:
                    r = p;
                    g = v;
                    b = t;
                    break;
                case 3:
                    r = p;
                    g = q;
                    b = v;
                    break;
                case 4:
                    r = t;
                    g = p;
                    b = v;
                    break;
                case 5:
                    r = v;
                    g = p;
                    b = q;
                    break;
            }
        }
        RED2(i,j)=r;
        GRN2(i,j)=g;
        BLU2(i,j)=b;
    }
}
"""
    inline(code, ['red','grn','blu','hue','sat','val'], local_dict=locals())
    return red,grn,blu






if __name__=='__main__' or __name__=='__mynamespace__':

    imagepath = 'images/mcgill/foliage_b/14.png'

    import Image
    import numpy

    from numpy.testing import assert_array_almost_equal

    R,G,B = Image.open(imagepath).split()

    R = numpy.array(R,dtype=numpy.int32)
    G = numpy.array(G,dtype=numpy.int32)
    B = numpy.array(B,dtype=numpy.int32)

    R/=255.0
    G/=255.0
    B/=255.0

    ## test rgb_to_hsv
    H,S,V = rgb_to_hsv_array_opt(R,G,B)
    h,s,v = rgb_to_hsv_array(R,G,B)
    assert_array_almost_equal(h,H,decimal=6)
    assert_array_almost_equal(s,S,decimal=6)
    assert_array_almost_equal(v,V,decimal=6)

    dp=6

    ## test hsv_to_rgb 
    R2,G2,B2 = hsv_to_rgb_array_opt(H,S,V)
    r2,g2,b2 = hsv_to_rgb_array(H,S,V)
    # test python implementations
    assert_array_almost_equal(r2,R,decimal=dp)
    assert_array_almost_equal(g2,G,decimal=dp)
    assert_array_almost_equal(b2,B,decimal=dp)
    # test C implementation
    assert_array_almost_equal(R2,R,decimal=dp)
    assert_array_almost_equal(G2,G,decimal=dp)
    assert_array_almost_equal(B2,B,decimal=dp)


    print "OK"






