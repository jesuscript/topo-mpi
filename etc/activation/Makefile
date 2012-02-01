# Makefile for the dot-product code

#
# C compiler and options
#
#CC=     /afs/inf.ed.ac.uk/user/s06/s0676011/CODE/topographica/bin/mpicc -O3 
#CC = mpicc -O3 
CC = cc -O3

LIB=    -lm

#
# Object files
#
OBJ=    activation.o\
	arralloc.o

#
# Compile
#
activation:	$(OBJ)
		$(CC) -o $@ $(OBJ) $(LIB)

.c.o:
	        $(CC) -c $<

#
# Clean out object files and the executable.
#
clean:
		rm *.o 
		rm activation
