#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>

void lin_grn_sim(int N, int Nrecord, double T, int Nt, double* X, double* x0,
                 double* W, double* mat){

  /* we simulate gene expression by alternating updates*/

  int s;
  /* Inititalize x */
  for(s=0;s<2*N;s++){
      X[s]=x0[s];
     }

  int i,j,k;
  double p;
  double dt = T/Nt;

  for(i=1;i<Nt;i++){
      printf("Time step: %d\n", i);
    for(j=0;j<2*N;j++){
      double dd = 0;
      for(k=0;k<2*N;k++){
        dd = dd + mat[j*(2*N)+k]*X[(i-1)*2*N+k];
      }

      double ddn = dd + W[i*2*N+j];
      //printf("%f,%f,%f\n",dd,ddn,dt*ddn);
      X[i*2*N+j] = X[(i-1)*2*N+j] + dt*ddn;

      if (X[i*2*N+j] < 0){
        X[i*2*N+j] = 0;
       }
      else if (X[i*2*N+j] > 1000){
        X[i*2*N+j] = 1000;
      }
     }
    }
  }

static PyObject* lin_grn(PyObject* Py_UNUSED(self), PyObject* args) {

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
  PyObject* _W = PyList_GetItem(list, 5);
  PyObject* _mat = PyList_GetItem(list, 6);


  double* X = malloc(2*N*Nt*sizeof(double));
  double* x0 = malloc(2*N*sizeof(double));
  double* W = malloc(2*N*Nt*sizeof(double));
  double* mat = malloc(4*N*N*sizeof(double));

  Py_ssize_t _x0_size = PyList_Size(_x0);
  for (Py_ssize_t j = 0; j < _x0_size; j++) {
  x0[j] = PyFloat_AsDouble(PyList_GetItem(_x0, j));
  if (PyErr_Occurred()) return NULL;
  }

  Py_ssize_t _W_size = PyList_Size(_W);
  for (Py_ssize_t j = 0; j < _W_size; j++) {
  W[j] = PyFloat_AsDouble(PyList_GetItem(_W, j));
  if (PyErr_Occurred()) return NULL;
  }

  Py_ssize_t _mat_size = PyList_Size(_mat);
  for (Py_ssize_t j = 0; j < _mat_size; j++) {
  mat[j] = PyFloat_AsDouble(PyList_GetItem(_mat, j));
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

  lin_grn_sim(N, Nrecord, T, Nt, X, x0, W, mat);

  npy_intp dims[2] = {Nt,2*N}; //row major order
  //Copy data into python list objects and free mem
  PyObject *X_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(X_out), X, 2*N*Nt*sizeof(double));

  free(X);
  free(x0);
  free(W);
  free(mat);

  return Py_BuildValue("O", X_out);

}
