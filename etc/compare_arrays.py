import sys
import numpy as np


if len(sys.argv)!=3:
    print "Error: provide 2 file names as arguments!"
else:
    a = np.genfromtxt(sys.argv[1])
    b = np.genfromtxt(sys.argv[2])
    np.testing.assert_array_almost_equal(a,b,7)
    print "Results matched"
