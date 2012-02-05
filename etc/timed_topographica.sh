#!/bin/bash



usage()
{
cat << EOF
usage: $0 options

This script runs and times topographica 


OPTIONS:
-c cortex density
-r retina density
-i number of iterations
-s script to run
-p number of processors
EOF
}


CORTEX=25
RETINA=25
ITERATIONS=10
SCRIPT=../examples/lissom_oo_or_mpi.ty
NCPU=2

while getopts "c:r:i:s:p:" OPTION
do
  case $OPTION in
      c)
	  CORTEX=$OPTARG
	  ;;
      r)
	  RETINA=$OPTARG
	  ;;
      i)
	  ITERATIONS=$OPTARG
	  ;;
      s)
	  SCRIPT=$OPTARG
	  ;;
      p)
	  NCPU=$OPTARG
      esac
done


echo "TOPOGRAPHICA RUNS"

echo ""
echo "### Serial Version ###"
../topographica -p cortex_density=$CORTEX -p retina_density=$RETINA -p lgn_density=$RETINA $SCRIPT -c "import timeit; print 'Serial:', timeit.Timer('topo.sim.run($ITERATIONS)','import topo').timeit(number=1)" -c "import numpy; numpy.savetxt('topo_serial_activity.out',topo.sim['V1'].activity)"

echo ""
echo "### Parallel Version ###"
../bin/mpiexec -n $NCPU ../topographica --mpi -p cortex_density=$CORTEX -p retina_density=$RETINA -p lgn_density=$RETINA -p mpi=True $SCRIPT -c "import timeit; print 'Parallel:', timeit.Timer('topo.sim.run($ITERATIONS)','import topo').timeit(number=1)" -c "import numpy; numpy.savetxt('topo_parallel_activity.out',topo.sim['V1'].activity)"

echo ""
echo "### Comparing results ###"
../bin/python compare_arrays.py topo_serial_activity.out topo_parallel_activity.out

#rm topo_serial_activity.out
#rm topo_parallel_activity.out
