/*
 * This file contains C routines for the MPI Casestudy.
 *
 * "datread" reads in an edges data file and can be called as follows:
 *
 *    float buf[M][N];
 *    datread("edge.dat", buf, M, N);
 *
 * "pgmwrite" writes an image as a PGM file, and can be called as follows:
 *
 *    float buf[M][N];
 *    pgmwrite("image.pgm", buf, M, N);
 *
 *  Note: you MUST link with the maths library -lm to access fabs, sqrt etc.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "arralloc.h"

/*
 *  Routine to read an "edges" data file into a 2D floating point array
 *  x[nx][ny]. Because of the way C handles (or fails to handle!)
 *  multi-dimensional arrays we have to cast the pointer to void.
 */

void datread(char *filename, void *vx, int nx, int ny)
{ 
  FILE *fp;

  int nxt, nyt, i, j, t;

  float *x = (float *) vx;

  if (NULL == (fp = fopen(filename,"r")))
  {
    fprintf(stderr, "datread: cannot open <%s>\n", filename);
    exit(-1);
  }

  fscanf(fp,"%d %d",&nxt,&nyt);

  if (nx != nxt || ny != nyt)
  {
    fprintf(stderr,
            "datread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
            nxt, nyt, nx, ny);
    exit(-1);
  }

  /*
   *  Must cope with the fact that the storage order of the data file
   *  is not the same as the storage of a C array, hence the pointer
   *  arithmetic to access x[i][j].
   */

  for (j=0; j<ny; j++)
  {
    for (i=0; i<nx; i++)
    {
      fscanf(fp,"%d", &t);
      x[(ny-j-1)+ny*i] = t;
    }
  }

  fclose(fp);
}


/*
 *  Routine to write a PGM image file from a 2D floating point array
 *  x[nx][ny]. Because of the way C handles (or fails to handle!)
 *  multi-dimensional arrays we have to cast the pointer to void.
 */

void pgmwrite(char *filename, void *vx, int nx, int ny)
{
  FILE *fp;

  int i, j, k, grey;

  float xmin, xmax, tmp;
  float thresh = 255.0;

  float *x = (float *) vx;

  if (NULL == (fp = fopen(filename,"w")))
  {
    fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
    exit(-1);
  }

  /*
   *  Find the max and min absolute values of the array
   */

  xmin = fabs(x[0]);
  xmax = fabs(x[0]);

  for (i=0; i < nx*ny; i++)
  {
    if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
    if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
  }

  fprintf(fp, "P2\n");
  fprintf(fp, "# Written by pgmwrite\n");
  fprintf(fp, "%d %d\n", nx, ny);
  fprintf(fp, "%d\n", (int) thresh);

  k = 0;

  for (j=ny-1; j >=0 ; j--)
  {
    for (i=0; i < nx; i++)
    {
      /*
       *  Access the value of x[i][j]
       */

      tmp = x[j+ny*i];

      /*
       *  Scale the value appropriately so it lies between 0 and thresh
       */

      if (xmin < 0 || xmax > thresh)
      {
        tmp = (int) ((thresh*((fabs(tmp-xmin))/(xmax-xmin))) + 0.5);
      }
      else
      {
        tmp = (int) (fabs(tmp) + 0.5);
      }

      /*
       *  Increase the contrast by boosting the lower values
       */
     
      grey = thresh * sqrt(tmp/thresh);

      fprintf(fp, "%3d ", grey);

      if (0 == (k+1)%16) fprintf(fp, "\n");

      k++;
    }
  }

  if (0 != k%16) fprintf(fp, "\n");
  fclose(fp);
}


int processImage(char *i_filename,char *o_filename, int nx, int ny){
  float** img;
  int i;

  img = arralloc(sizeof(float), 2, nx, ny);
  
  datread(i_filename, img, nx,ny);

  float** edged = arralloc(sizeof(float),2,nx+2,ny+2);
  
  for(i=0;i<nx+2;i++)
    for(j=0;j<ny+2;j++){
      if(j==0 || i==0 || j==ny+1 || i==nx+1)
	edged[i][j]=0;
      else
	edged[i][j]=
    }


  pgmwrite(o_filename, img, nx,ny);

  return 0;
}

int main(int argc, char **argv ){
    return processImage("edge192x360.dat","edge192x360.pgm",192,360);
    
}

