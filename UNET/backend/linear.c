#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>

void LinearSim(int N, int Nrecord, double T, int Nt, double* X, double* Y, 
				    double* x0, double* y0, double* noise_x, double* noise_y, double* mat, 
				    double a, double b, double c){

  /* we simulate gene expression by alternating 
     updates between protein and RNA*/

  int s;
  /* Inititalize x */
  for(s=0;s<N;s++){
      X[s]=x0[s];
      Y[s]=y0[s];
     }

  int i,j,k;
  double p;
  double dt = T/Nt;

  for(i=1;i<Nt;i++){
    //printf("Time step: %d\n", i);
    
    //update protein
    for(j=0;j<N;j++){
      double dp = a*X[(i-1)*N+j] - b*Y[(i-1)*N+j];
      dp = dp + noise_y[i*N+j];
      Y[i*N+j] = Y[(i-1)*N+j] + dt*dp;

	  //enforce bounds
      if (Y[i*N+j] < 0){
        Y[i*N+j] = 0;
       }
     }
     
    //update RNA 
    for(j=0;j<N;j++){
      double dr = 0;
      for(k=0;k<N;k++){
        dr = dr + mat[j*N+k]*Y[(i-1)*N+k];
      }
	  dr = dr - c*X[(i-1)*N+j] + noise_x[i*N+j];
      X[i*N+j] = X[(i-1)*N+j] + dt*dr;

      if (X[i*N+j] < 0){
        X[i*N+j] = 0;
       }
     }
     
     
    }
  }

static PyObject* Linear(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

  //Quantities that will be passed to the simulation directly
  int N = PyFloat_AsDouble(PyList_GetItem(list, 0));
  int Nrecord = PyFloat_AsDouble(PyList_GetItem(list, 1));
  float T = PyFloat_AsDouble(PyList_GetItem(list, 2));
  int Nt = PyFloat_AsDouble(PyList_GetItem(list, 3));

  //Chunks of memory passed to the function as pointers
  PyObject* _x0 = PyList_GetItem(list, 4);
  PyObject* _y0 = PyList_GetItem(list, 5);
  PyObject* _mat = PyList_GetItem(list, 6);
  PyObject* _noise_x = PyList_GetItem(list, 7);
  PyObject* _noise_y = PyList_GetItem(list, 8);
  double a = PyFloat_AsDouble(PyList_GetItem(list, 9)); //translation rate
  double b = PyFloat_AsDouble(PyList_GetItem(list, 10)); //protein degradation rate
  double c = PyFloat_AsDouble(PyList_GetItem(list, 11)); //RNA degradation rate
    
  double* X = malloc(N*Nt*sizeof(double));
  double* Y = malloc(N*Nt*sizeof(double));
  double* x0 = malloc(N*sizeof(double));
  double* y0 = malloc(N*sizeof(double));
  double* mat = malloc(N*N*sizeof(double));
  double* noise_x = malloc(N*Nt*sizeof(double));
  double* noise_y = malloc(N*Nt*sizeof(double));

  Py_ssize_t _x0_size = PyList_Size(_x0);
  for (Py_ssize_t j = 0; j < _x0_size; j++) {
  x0[j] = PyFloat_AsDouble(PyList_GetItem(_x0, j));
  if (PyErr_Occurred()) return NULL;
  }

  Py_ssize_t _y0_size = PyList_Size(_y0);
  for (Py_ssize_t j = 0; j < _y0_size; j++) {
  y0[j] = PyFloat_AsDouble(PyList_GetItem(_y0, j));
  if (PyErr_Occurred()) return NULL;
  }

  Py_ssize_t _mat_size = PyList_Size(_mat);
  for (Py_ssize_t j = 0; j < _mat_size; j++) {
  mat[j] = PyFloat_AsDouble(PyList_GetItem(_mat, j));
  if (PyErr_Occurred()) return NULL;
  }
  
  Py_ssize_t _noise_x_size = PyList_Size(_noise_x);
  for (Py_ssize_t j = 0; j < _noise_x_size; j++) {
  noise_x[j] = PyFloat_AsDouble(PyList_GetItem(_noise_x, j));
  if (PyErr_Occurred()) return NULL;
  }
  
   Py_ssize_t _noise_y_size = PyList_Size(_noise_y);
  for (Py_ssize_t j = 0; j < _noise_y_size; j++) {
  noise_y[j] = PyFloat_AsDouble(PyList_GetItem(_noise_y, j));
  if (PyErr_Occurred()) return NULL;
  }

  //Print params
  printf("\n\n###################\n");
  printf("Parameters:\n\n");
  printf("N = %d\n", N);
  printf("Nrecord = %d\n", Nrecord);
  printf("T = %f\n", T);
  printf("Nt = %i\n", Nt);
  printf("###################\n\n");

  LinearSim(N, Nrecord, T, Nt, X, Y, x0, y0, noise_x, noise_y, mat, a, b, c);
  printf("%f,%f\n",X[100],Y[100]);

  free(x0);
  free(y0);
  free(mat);
  free(noise_x);
  free(noise_y);

  npy_intp dims[2] = {Nt,N}; //row major order
  //Copy data into python list objects and free mem
  PyObject *X_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(X_out), X, N*Nt*sizeof(double));
  free(X);

  PyObject *Y_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(Y_out), Y, N*Nt*sizeof(double));
  free(Y);

  return Py_BuildValue("(OO)", X_out, Y_out);

}
