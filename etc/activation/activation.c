#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <string.h>


#include "arralloc.h"

#define STRENGTH1 1.0
#define STRENGTH2 0.25
#define ACT_APPROX 0.5
#define TIME_CONSTANT 0.3

//define which versions of activation to use
#define SEQ 0
#define MPI 1
#define MPI2 0 // do not use (removed routine calls from main())


#define PRINT 0 // print timings to the command line


//>>>>> Timing vars >>>>>
//>>> main >>>
double total_run_s, total_run_e;
double utils_main_s, utils_main_t;
double create_cfs_s,create_cfs_e;
double act_seq_s,act_seq_e;
double act_mpi_s,act_mpi_e;
double compare_results_s, compare_results_e;
//>>> act_seq >>>
double total_act_seq_s, total_act_seq_e;
double dp_seq_s, dp_seq_t;
double utils_seq_s, utils_seq_t;
double of_seq_s, of_seq_t;
//>>> act_mpi >>>
double total_act_mpi_s, total_act_mpi_e;
double dp_mpi_s, dp_mpi_t;
double utils_mpi_s, utils_mpi_t;
double distr_weights_s, distr_weights_e;
double of_mpi_s, of_mpi_t;
double bc_mpi_s, bc_mpi_t;
double gather_mpi_s, gather_mpi_t;
//<<<<<<<<<<<<<<<<<<<<<<<





//>>>>>>>>>>>>>>>>>>>> I/O >>>>>>>>>>>>>>>>>>>>


void printActivity(float* activity, int m, int n, int size){
  int i,j;
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      if((i*n + j) >= size){printf("\n");return;}
      printf("%.8f ",activity[i*n + j]);
    }
    printf("\n");
  }
}

void fprintActivity(float* activity, int m, int n, int size){
  int i,j;
  FILE *f;
  f= fopen("c_activity.out","w");

  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      if((i*n + j) >= size){fprintf(f,"\n");return;}
      fprintf(f, "%.9f ",activity[i*n + j]);
    }
    fprintf(f,"\n");
  }

  fclose(f);
}


void printCFs(float ***flatcfs,int m, int n, int size){
  int i,j,k;
  for(i=0;i<size;i++){
    if(i!=0)
      printf("-------------------------------------\n");
    for(j=0;j<m;j++){
      for(k=0;k<n;k++){
	printf("%.7f\t", flatcfs[i][j][k]);
      }
      printf("\n");
    }
  }
}

void printCF(float ***flatcfs, int index, int m, int n){
  int i,j;
  for(i=0;i<m;i++){
    for(j=0;j<n;j++)
      printf("%.10f\t",flatcfs[index][i][j]);
    printf("\n");
  }
}


//not mpi-friendly
void fprintCFs(float ***flatcfs,int m, int n){
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);


  if(rank!=0)
    return;

  int i,j,k;

  FILE *f;
  f= fopen("cfs.out","w");

  for(i=0;i<m*n;i++){
    fprintf(f,":%i\n",i);
    for(j=0;j<m;j++){
      for(k=0;k<n;k++){
	fprintf(f,"%f", flatcfs[i][j][k]);
      }
      fprintf(f,"\n");
    }
  }

  fclose(f);
}





//>>>>>>>>>>>>>>>>>>>> Array Generators >>>>>>>>>>>>>>>>>>>>

void initInputActivity(float *input_activity, int m, int n){
  int i;

  for(i=0;i<m*n;i++)
    input_activity[i]=(float) (rand() % 1000000)/1000000; 
}


void createCFs(float ***flatcfs, int m, int n){
  //srand ( time(NULL) );

  int i,j,k;
  int tot_size = m*n;
  

  for(i=0;i<m*n;i++){
    for(j=0;j<m;j++)
      for(k=0;k<n;k++){
	//flatcfs[i][j][k] = (float) (rand() % 1000000)/1000000;
	flatcfs[i][j][k] = (float)1 + i + (float)(n*j + k) / tot_size; 
      }
  }
}



// creates connection fields locally to avoid broadcasting
void MPIcreateCFs(float ***local_fcfs, int start, int end, int m, int n){
  int i,j,k;
  int tot_size = m*n;
  for(i=0;i<(end-start);i++)
    for(j=0;j<m;j++)
      for(k=0;k<n;k++){
	local_fcfs[i][j][k]=(float)1 + i+start + (float)(n*j + k) / tot_size; 
      }
}


void movingDot(float* input_activity, int pos, int m, int n){
  int i;

  int sz = m*n;
  pos %= sz;

  for(i=0;i<sz;i++){
    input_activity[i] = (float) (i==pos)*STRENGTH1;
    if(i==(sz-pos-1))
      input_activity[i] = STRENGTH2;
  }
}


void zeros(float* activity, int size){
  int i=0;
  for(;i<size;i++)
    activity[i] = 0.0;
}



//>>>>>>>>>>>>>>>>>>>> Response Functions >>>>>>>>>>>>>>>>>>>>


void dotProduct(float ***flatcfs, float* activity,float* input_activity, \
		int activity_chunk_size, float strength, int m, int n){
  int i,j,k;
  float tot;


  for(i=0;i<activity_chunk_size;i++){
    tot=0;
    for(j=0;j<m;j++)
      for(k=0;k<n;k++){
	tot+=flatcfs[i][j][k]*input_activity[j*n + k];
      }
    activity[i]=tot*strength;
  }

}

/*
void dotProductMPI_2(float ***flatcfs, float* activity, float* input_activity, 
		     int exchange_chunk_size, int m){
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Request *r_request, s_request;
  MPI_Status *r_status;
  int rank,size;
  int units_pn;
  int i,j,k;
  int matrix_size = m*m;
  int local_top;

  
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);
  
  units_pn = matrix_size / size;

  //setting up receives

  int n_recvs = 0;

  for(i=0;i<size;i++)
    if(i!=rank)
      if(i!=size-1){

	n_recvs += ceil((float)units_pn/exchange_chunk_size);
      }
      else{

	n_recvs += ceil((float)(matrix_size - units_pn*i) / exchange_chunk_size);
      }

  r_request = (MPI_Request*)malloc(sizeof(MPI_Request) * n_recvs);

  // TODO: try getting rid of status array and use single variable instead
  r_status = (MPI_Status*)malloc(sizeof(MPI_Status) * n_recvs); 
  
 
  k=0;
  for(i=0;i<size;i++)
    if(i!=rank){
      if(i!=size-1)
	local_top = (i+1) * units_pn;
      else
	local_top = matrix_size;
      j = units_pn * i;
      while(j<local_top){

	if (j+exchange_chunk_size < local_top){
	  MPI_Irecv(&activity[j], exchange_chunk_size, MPI_FLOAT, i, j, comm, &r_request[k]);
	}
	else{
	  MPI_Irecv(&activity[j], local_top - j, MPI_FLOAT, i, j, comm, &r_request[k]);
	}
	
	j+=exchange_chunk_size;
	k++;
      }
    }

  float tot;
  int local_u_num;
  int chunk_start=0;

  if(rank!=size-1)
    local_u_num = units_pn;
  else
    local_u_num = matrix_size - units_pn*(size-1);

  // main loop


  for(i=0; i<local_u_num;){
    tot = 0;
    for(j=0;j<m;j++)
      for(k=0;k<m;k++){
	tot += flatcfs[i][j][k]*input_activity[j*m + k];
      }


    activity[i + rank*units_pn]=tot;
    i++;

    if ((i == local_u_num) || (i - chunk_start == exchange_chunk_size)){
      k = units_pn*rank + chunk_start;
      for(j=0;j<size;j++)
	if(j!=rank)
	  MPI_Isend(&activity[k], i - chunk_start, MPI_FLOAT, j, k,comm,&s_request);	
      chunk_start = i;

    }

  }


  MPI_Waitall(n_recvs,r_request,r_status);


}
*/







//>>>>>>>>>>>>>>>>>>>> Output Functions >>>>>>>>>>>>>>>>>>>>

void normalise(float *activity, int m, int n){
  int i;
  float tot;

  tot=0.0f;

  for(i=0;i<m*n;i++)
    tot+= activity[i];
  for(i=0;i<m*n;i++)
    activity[i] /= tot;
    
}



void normaliseCFs(float ***flatcfs, int cf_num, int m, int n){
  int i,j,k;
  float tot;
  
  for(i=0;i<cf_num;i++){
    tot=0;
    for(j=0;j<m;j++)
      for(k=0;k<n;k++)
	tot+=flatcfs[i][j][k];
    for(j=0;j<m;j++)
      for(k=0;k<n;k++)
	flatcfs[i][j][k] /=tot;

  }
    
}


void hysteresis(float *new_activity, float *old_activity, int act_size){
  int i;

  for(i=0;i<act_size;i++)
    new_activity[i] = old_activity[i] + (new_activity[i] - old_activity[i])*TIME_CONSTANT;
}


//>>>>>>>>>>>>>>>>>>>> Array Utils >>>>>>>>>>>>>>>>>>>>


void copyActivity(float *from, float *to, int m, int n){
  int i;
  for(i=0;i<m*n;i++){
    to[i] = from[i];
  }
}

int compareActivity(float *act_1, float *act_2, float eps, int m, int n){
  int i;
  for(i=0;i<m*n;i++){
    if (fabs(act_1[i] - act_2[i]) > eps)
      return 1;
  }
  return 0;
}


void extractFCFChunk(float ***flatcfs, float ***local_fcfs, int start, int end){
  int i;
  for(i=0;i<(end-start);i++)
    local_fcfs[i]=flatcfs[start+i];
}







//>>>>>>>>>>>>>>>>>>>> Activation >>>>>>>>>>>>>>>>>>>>


void activateSeq(float ***flatcfs, float *activity, float strength, int m, int n, int its ){
  //timing shizzle
  total_act_seq_s = MPI_Wtime();
  utils_seq_s = MPI_Wtime();
  dp_seq_t=0;
  of_seq_t=0;

  float *local_ia, *old_a;

  int i;
  

  local_ia = (float*) malloc (sizeof(float) * m * n);
  old_a = (float*) malloc(sizeof(float) * m * n);

  zeros(old_a,m*n);
  utils_seq_t = MPI_Wtime() - utils_seq_s;


  for(i=0;i<its;i++){
    utils_seq_s = MPI_Wtime();
    movingDot(local_ia,i,m,n);
    utils_seq_t += MPI_Wtime() - utils_seq_s;

    dp_seq_s = MPI_Wtime();
    dotProduct(flatcfs,activity,local_ia,m*n,strength,m,n);
    dp_seq_t += MPI_Wtime() - dp_seq_s;

    of_seq_s = MPI_Wtime();
    hysteresis(activity,old_a,m*n);
    of_seq_t +=MPI_Wtime() - of_seq_s;

    utils_seq_s = MPI_Wtime();
    copyActivity(activity,old_a,m,n);
    utils_seq_t += MPI_Wtime() - utils_seq_s;
  }

  total_act_seq_e = MPI_Wtime();
}


void activateMPI_1(float ***flatcfs, float *activity, float strength, int m, int n, int its){
  //timing stuff
  utils_mpi_t = 0;
  of_mpi_t = 0;
  dp_mpi_t = 0;
  bc_mpi_t = 0;
  gather_mpi_t = 0;

  total_act_mpi_s = MPI_Wtime();
  utils_mpi_s = MPI_Wtime();


  MPI_Comm comm;
  int rank, size;
  int units_pn; // units (cf's, activities) per node
  int local_u_num;
  int i,j;

  float ***local_fcfs; // local flatcfs
  float *local_a, *local_ia, *old_a; // local activity and input activity

  comm = MPI_COMM_WORLD;
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  units_pn = m*n / size;

  if(rank!=size-1)
    local_u_num = units_pn;
  else
    local_u_num = m*n - rank*units_pn;
  
  local_fcfs = arralloc(sizeof(float),3,local_u_num,m,n);
  local_a = (float*) malloc(sizeof(float) * local_u_num);
  local_ia = (float*) malloc(sizeof(float) * m*n);
  if(rank==0)
    old_a = (float*) malloc(sizeof(float) * m*n);


  // creating displacements and rcounts arrays later used in the Gatherv routine


  int* disps; 
  int* rcounts; 


  if (rank==0){
    disps = (int*) malloc(sizeof(int) * size);
    rcounts = (int*)malloc(sizeof(int) * size);
    for(i=0;i<size;i++){
      if(i!=size-1){
	rcounts[i]=units_pn;
	disps[i]=(i * units_pn );
      }
      else{
	rcounts[i]=m*n - i*units_pn;
	disps[i]=(i * units_pn );
      }
    }
  }
  utils_mpi_t += MPI_Wtime() - utils_mpi_s;


  distr_weights_s = MPI_Wtime();
  //distributing connection fields
  //MPI_Bcast(&(flatcfs[0][0][0]),m*n * m * n,MPI_FLOAT,0,comm);
  //extractFCFChunk(flatcfs, local_fcfs, rank*units_pn, rank*units_pn + local_u_num);
  MPIcreateCFs(local_fcfs,rank*units_pn, rank*units_pn + local_u_num,m,n);
  normaliseCFs(local_fcfs, local_u_num, m, n);
  distr_weights_e = MPI_Wtime();

  utils_mpi_s = MPI_Wtime();
  if(rank==0)
    zeros(old_a,m*n);
  utils_mpi_t += MPI_Wtime() - utils_mpi_s;

  //Main loop


  for(i=0;i<its;i++){
    utils_mpi_s = MPI_Wtime();
    if (rank==0)
      movingDot(local_ia,i,m,n);
    utils_mpi_t += MPI_Wtime() - utils_mpi_s;
    
    bc_mpi_s = MPI_Wtime();
    MPI_Bcast(local_ia, m*n, MPI_FLOAT, 0, comm);
    bc_mpi_t+= MPI_Wtime() - bc_mpi_s;

    dp_mpi_s = MPI_Wtime();
    dotProduct(local_fcfs, local_a, local_ia, local_u_num, strength,m,n);
    dp_mpi_t += MPI_Wtime() - dp_mpi_s;
    
    gather_mpi_s = MPI_Wtime();
    MPI_Gatherv(local_a,local_u_num,MPI_FLOAT, activity, rcounts, disps, MPI_FLOAT, 0, comm);
    gather_mpi_t += MPI_Wtime() - gather_mpi_s;
    
    of_mpi_s = MPI_Wtime();
    if (rank==0){
      hysteresis(activity,old_a,m*n);
    }
    of_mpi_t += MPI_Wtime() - of_mpi_s;

    utils_mpi_s = MPI_Wtime();
    if (rank==0){
      copyActivity(activity,old_a,m,n);
    }
    utils_mpi_t += MPI_Wtime() - utils_mpi_s;
      
  } 

  total_act_mpi_e = MPI_Wtime();
}


/*
void activateMPI_2(float ***flatcfs, float *activity, float* input_activity, int m, int its, int exchange_chunk_size){
  MPI_Comm comm;
  int rank, size;
  int units_pn;
  int local_u_num;
  int i,j;

  float ***local_fcfs;
  float *local_ia;

  //for timings

  double end_t, start_t;


  comm = MPI_COMM_WORLD;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);

  units_pn = m*m / size;

  if(rank!=size-1)
    local_u_num = units_pn;
  else
    local_u_num = m*m - rank*units_pn;
  
  local_fcfs = arralloc(sizeof(float),3,local_u_num,m,m);
  local_ia = (float*) malloc(sizeof(float) * m*m);

  copyActivity(input_activity, local_ia,m);

  //distributing connection fields
  MPI_Bcast(&(flatcfs[0][0][0]),m*m * m * m,MPI_FLOAT,0,comm);
  extractFCFChunk(flatcfs, local_fcfs, rank*units_pn, rank*units_pn + local_u_num);

  MPI_Bcast(local_ia, m*m, MPI_FLOAT, 0, comm);


  if(rank==0) start_t = MPI_Wtime();
  for(i=0;i<its;i++){
    dotProductMPI_2(local_fcfs,activity,local_ia,exchange_chunk_size,m);

    normalise(activity,m);

    copyActivity(activity,local_ia,m);
  }
  if(rank==0) end_t = MPI_Wtime();


  if(rank == 0){
    printf("\n");
    printf("=== Parallel v.2 Time ===\n");
    printf("more detailed timings will come later\n");
    printf("-------------------------\n");
    printf("Total:\t\t%g\n",end_t - start_t);
  }

}
*/
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





int main(int argc, char **argv){
  MPI_Init(&argc,&argv);  

  total_run_s = MPI_Wtime();  
  utils_main_t = 0;
  utils_main_s = MPI_Wtime();

  int rank,size;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  if(argc<4 || argc>5){
    if (rank==0)
      printf("Arguments: <M>, <N>, <Iterations> [filename]\n");
    return -1;
  }



  float ***flatcfs;
  float *activity, *input_activity;

  int m = atoi(argv[1]);
  int n = atoi(argv[2]);
  int its = atoi(argv[3]);

  float strength = ACT_APPROX * (m*n) / (STRENGTH1+STRENGTH2);


  activity = (float*) malloc(m*n*sizeof(float));  
  flatcfs = arralloc(sizeof(float),3,m*n,m,n);
  input_activity = (float*) malloc(m*n*sizeof(float));



  //used for comparing results of computations using different algorithms 
  //and testing for correctness
  float *activity_seq, *activity_mpi1, *activity_mpi2;
  activity_seq = (float*) malloc(sizeof(float) * m*n);
  activity_mpi1 = (float*) malloc(sizeof(float) * m*n);


  utils_main_t += MPI_Wtime() - utils_main_s;

  create_cfs_s = MPI_Wtime();
  if(rank == 0){  
    createCFs(flatcfs,m,n);
    normaliseCFs(flatcfs,m*n,m,n);
  }
  create_cfs_e = MPI_Wtime();


  act_seq_s = MPI_Wtime();
  if(rank ==0 && (SEQ || size==1))
    activateSeq(flatcfs, activity,strength, m,n, its);
  act_seq_e = MPI_Wtime();



  utils_main_s = MPI_Wtime();
  if(rank==0 && (SEQ || size==1)){
    copyActivity(activity,activity_seq,m,n);
    fprintActivity(activity, m,n, m*n);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  utils_main_t += MPI_Wtime() - utils_main_s;



  act_mpi_s = MPI_Wtime();
  activateMPI_1(flatcfs, activity, strength, m,n, its);
  act_mpi_e = MPI_Wtime();

  utils_main_s = MPI_Wtime();
  if(rank==0 && MPI){
    copyActivity(activity,activity_mpi1,m,n);
    if(!(SEQ || size==1))
      fprintActivity(activity, m,n, m*n);
  }
  utils_main_t += MPI_Wtime() - utils_main_s;

  // compare resulting arrays if both mpi and serial were turned on


  utils_main_s=MPI_Wtime();
  if(rank==0 && MPI && (SEQ || size==1)){
    if (compareActivity(activity_seq,activity_mpi1,0.00000001f,m,n)!=0){
      printf("Activity mismatch in MPI 1!\n");
      printActivity(activity_seq,m,n,m*n);
      printf("\n");
      printActivity(activity_mpi1,m,n,m*n);
    }else if(PRINT){
      printf("Serial and MPI activities match (bare C)\n\n");
    }
  }
  utils_main_t += MPI_Wtime() - utils_main_s;

  




  total_run_e = MPI_Wtime();

  //labels;
  char total_run_l[] = "main: Total";
  char utils_main_l[] = "main: Utils";
  char create_cfs_l[] = "main: Creating CFs";
  char act_seq_l[] = "main: Serial Activate";
  char act_mpi_l[] = "main: Parallel Activate";
  char total_act_seq_l[] = "serial: Total";
  char utils_seq_l[] = "serial: Utils";
  char dp_seq_l[] = "serial: Dot Product";
  char of_seq_l[] = "serial: Output Function";
  char total_act_mpi_l[] = "parallel: Total";
  char utils_mpi_l[] = "parallel: Utils";
  char distr_weights_l[] = "parallel: Distributing Weights";
  char bc_mpi_l[] = "parallel: Broadcasting Inputs";
  char dp_mpi_l[] = "parallel: Dot Product";
  char gather_mpi_l[] = "parallel: Gathering Outputs";
  char of_mpi_l[]= "parallel: Output Function";


  //Timings output
  if(rank==0 && PRINT){
    printf("========================== main() ==========================\n");
    printf("%s\t\t\t\t\t%.4f\n",total_run_l,total_run_e-total_run_s);
    printf("------------------------------------------------------------\n");
    printf("%s\t\t\t\t\t%.4f\n",utils_main_l,utils_main_t);
    printf("%s\t\t\t\t%.4f\n",create_cfs_l,create_cfs_e-create_cfs_s);
    printf("%s\t\t\t\t%.4f\n",act_seq_l,act_seq_e-act_seq_s);
    printf("%s\t\t\t\t%.4f\n",act_mpi_l,act_mpi_e-act_mpi_s);
    if((SEQ || size==1)){
      printf("======================= activateSeq() ======================\n");
      printf("%s\t\t\t\t\t%.4f\n",total_act_seq_l,total_act_seq_e - total_act_seq_s);
      printf("------------------------------------------------------------\n");
      printf("%s\t\t\t\t\t%.4f\n",utils_seq_l,utils_seq_t);
      printf("%s\t\t\t\t%.4f\n",dp_seq_l,dp_seq_t);
      printf("%s\t\t\t\t%.4f\n",of_seq_l,of_seq_t);
    }
    if(MPI){
      printf("======================= activateMPI() ======================\n");
      printf("%s\t\t\t\t\t%.4f\n",total_act_mpi_l,total_act_mpi_e - total_act_mpi_s);
      printf("------------------------------------------------------------\n");
      printf("%s\t\t\t\t\t%.4f\n",utils_mpi_l,utils_mpi_t);
      printf("%s\t\t\t%.4f\n",distr_weights_l,distr_weights_e - distr_weights_s);
      printf("%s\t\t\t%.4f\n",bc_mpi_l,bc_mpi_t);
      printf("%s\t\t\t\t%.4f\n",dp_mpi_l,dp_mpi_t);
      printf("%s\t\t\t%.4f\n",gather_mpi_l,gather_mpi_t);
      printf("%s\t\t\t%.4f\n",of_mpi_l,of_mpi_t);

    }
    printf("============================================================\n");
  }


  if(argc==5 && rank==0){
    char* indent = "  "; //xml indentation
    
    FILE *f;
    f = fopen(argv[4],"a");//filename as command line argument

    fseek(f, 0, SEEK_END); //
    if (ftell(f) == 0)//new file, add opening root xml tag
      fprintf(f,"<runs>\n");
    else{ // file wasn't empty, delete the closing root xml tag
      fclose(f);

      char tmp_str[100];
      FILE *temp;

      f = fopen(argv[4],"r");
      temp = fopen("timings.tmp.xml","wb");
      
      while(fgets(tmp_str,100,f) != NULL){
	if(strstr(tmp_str,"</runs>"))
	  break;
	else
	  fputs(tmp_str,temp);
      }

      fclose(f);
      fclose(temp);
      char * command = //concatenating strings to build up a unix command
	malloc(snprintf(NULL, 0, "mv timings.tmp.xml %s", argv[4]) + 1);
      sprintf(command,"mv timings.tmp.xml %s",argv[4]);
      if(system(command))
	printf("something's wrong with the mv system command");
      f = fopen(argv[4],"a");
    }

    // ... and this is why I hate C sometimes
    


    // print run params: size, m, n, i
    fprintf(f,"%s<run cpu_cores=\'%i\' density=\'%i\' m=\'%i\' n=\'%i\' iterations=\'%i\'>\n",indent,size, m*n, m, n, its);
    indent = "    ";

    fprintf(f,"%s<main_tot name=\'%s\'>%.8f</main_tot>\n",indent,total_run_l,total_run_e-total_run_s);
    fprintf(f,"%s<main_utils name=\'%s\'>%.8f</main_utils>\n",indent,utils_main_l,utils_main_t);
    fprintf(f,"%s<main_create_cfs name=\'%s\'>%.8f</main_create_cfs>\n",indent,create_cfs_l,create_cfs_e-create_cfs_s);
    fprintf(f,"%s<main_seq_act name=\'%s\'>%.8f</main_seq_act>\n",indent,act_seq_l,act_seq_e-act_seq_s);
    fprintf(f,"%s<main_mpi_act name=\'%s\'>%.8f</main_mpi_act>\n",indent,act_mpi_l,act_mpi_e-act_mpi_s);


    if((SEQ || size==1)){
      fprintf(f,"%s<act_seq_tot name=\'%s\'>%.8f</act_seq_tot>\n",indent,total_act_seq_l,total_act_seq_e - total_act_seq_s);
      fprintf(f,"%s<act_seq_utils name=\'%s\'>%.8f</act_seq_utils>\n",indent,utils_seq_l,utils_seq_t);
      fprintf(f,"%s<act_seq_dp name=\'%s\'>%.8f</act_seq_dp>\n",indent,dp_seq_l,dp_seq_t);
      fprintf(f,"%s<act_seq_of name=\'%s\'>%.8f</act_seq_of>\n",indent,of_seq_l,of_seq_t);
    }
    
    if(MPI){
      fprintf(f,"%s<act_mpi_tot name=\'%s\'>%.8f</act_mpi_tot>\n",indent,total_act_mpi_l,total_act_mpi_e - total_act_mpi_s);
      fprintf(f,"%s<act_mpi_utils name=\'%s\'>%.8f</act_mpi_utils>\n",indent,utils_mpi_l,utils_mpi_t);
      fprintf(f,"%s<act_mpi_distr_weights name=\'%s\'>%.8f</act_mpi_distr_weights>\n",indent, 
	      distr_weights_l,distr_weights_e - distr_weights_s);
      fprintf(f,"%s<act_mpi_bc name=\'%s\'>%.8f</act_mpi_bc>\n",indent,bc_mpi_l,bc_mpi_t);
      fprintf(f,"%s<act_mpi_dp name=\'%s\'>%.8f</act_mpi_dp>\n",indent,dp_mpi_l,dp_mpi_t);
      fprintf(f,"%s<act_mpi_gather name=\'%s\'>%.8f</act_mpi_gather>\n",indent,gather_mpi_l,gather_mpi_t);
      fprintf(f,"%s<act_mpi_of name=\'%s\'>%.8f</act_mpi_of>\n",indent,of_mpi_l,of_mpi_t);
    }
    indent = "  ";
    fprintf(f,"%s</run>\n",indent);
    fprintf(f,"</runs>\n");

    fclose(f);
  }



  MPI_Finalize();
  return 0;
}
