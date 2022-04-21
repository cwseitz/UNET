#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>

void HillSim(int N, int Nrecord, double T, int Nt,
                 double* X, double* x0, double* w1, double* w2,
                 double* h, double* K, double* b, double* lam, double* q, double* n){

  int s;
  /* Inititalize x */
  for(s=0;s<N;s++){
      X[s]=x0[s];
     }

  int i,j,k;
  double p;

  for(i=1;i<Nt;i++){
      printf("Time step: %d\n", i);
    for(j=0;j<N;j++){
      p = b[j];
      for(k=0;k<N;k++){
        p += K[j*N+k]*(pow(X[(i-1)*N+k],n[j*N+k]))/(pow(X[(i-1)*N+k],n[j*N+k]) + pow(h[j*N+k],n[j*N+k]));
      }
      p = abs(p);
      X[i*N+j] = X[(i-1)*N+j] + p - lam[j]*X[(i-1)*N+j] + q[j]*(sqrt(p)*w1[i*N+j] + sqrt(lam[j]*X[(i-1)*N+j])*w2[i*N+j]);
      if (X[i*N+j] < 0){
        X[i*N+j] = 0;
      }
    }
  }
}

static PyObject* Hill(PyObject* Py_UNUSED(self), PyObject* args) {

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
    PyObject* _w1 = PyList_GetItem(list, 5);
    PyObject* _w2 = PyList_GetItem(list, 6);
    PyObject* _h = PyList_GetItem(list, 7);
    PyObject* _K = PyList_GetItem(list, 8);
    PyObject* _b = PyList_GetItem(list, 9);
    PyObject* _lam = PyList_GetItem(list, 10);
    PyObject* _q = PyList_GetItem(list, 11);
    PyObject* _n = PyList_GetItem(list, 12);

    double* X = malloc(N*Nt*sizeof(double));
    double* x0 = malloc(N*sizeof(double));
    double* w1 = malloc(N*Nt*sizeof(double));
    double* w2 = malloc(N*Nt*sizeof(double));
    double* h = malloc(N*N*sizeof(double));
    double* K = malloc(N*N*sizeof(double));
    double* b = malloc(N*sizeof(double));
    double* lam = malloc(N*sizeof(double));
    double* q = malloc(N*sizeof(double));
    double* n = malloc(N*N*sizeof(double));

    Py_ssize_t _x0_size = PyList_Size(_x0);
    for (Py_ssize_t j = 0; j < _x0_size; j++) {
      x0[j] = PyFloat_AsDouble(PyList_GetItem(_x0, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _w1_size = PyList_Size(_w1);
    for (Py_ssize_t j = 0; j < _w1_size; j++) {
      w1[j] = PyFloat_AsDouble(PyList_GetItem(_w1, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _w2_size = PyList_Size(_w2);
    for (Py_ssize_t j = 0; j < _w2_size; j++) {
      w2[j] = PyFloat_AsDouble(PyList_GetItem(_w2, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _h_size = PyList_Size(_h);
    for (Py_ssize_t j = 0; j < _h_size; j++) {
      h[j] = PyFloat_AsDouble(PyList_GetItem(_h, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _K_size = PyList_Size(_K);
    for (Py_ssize_t j = 0; j < _K_size; j++) {
      K[j] = PyFloat_AsDouble(PyList_GetItem(_K, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _b_size = PyList_Size(_b);
    for (Py_ssize_t j = 0; j < _b_size; j++) {
      b[j] = PyFloat_AsDouble(PyList_GetItem(_b, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _lam_size = PyList_Size(_lam);
    for (Py_ssize_t j = 0; j < _lam_size; j++) {
      lam[j] = PyFloat_AsDouble(PyList_GetItem(_lam, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _q_size = PyList_Size(_q);
    for (Py_ssize_t j = 0; j < _q_size; j++) {
      q[j] = PyFloat_AsDouble(PyList_GetItem(_q, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _n_size = PyList_Size(_n);
    for (Py_ssize_t j = 0; j < _n_size; j++) {
      n[j] = PyFloat_AsDouble(PyList_GetItem(_n, j));
      if (PyErr_Occurred()) return NULL;
    }

    //Print params
    printf("\n\n###################\n");
    printf("Parameters:\n\n");
    printf("N = %d\n", N);
    printf("Nrecord = %d\n", Nrecord);
    printf("T = %f\n", T);
    printf("Nt = %f\n", Nt);
    printf("###################\n\n");

    HillSim(N, Nrecord, T, Nt, X, x0, w1, w2, h, K, b, lam, q, n);

  npy_intp dims[2] = {Nt,N}; //row major order
  //Copy data into python list objects and free mem
  PyObject *X_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(X_out), X, N*Nt*sizeof(double));

  free(X);
  free(x0);
  free(w1);
  free(w2);
  free(h);
  free(K);
  free(b);
  free(lam);
  free(q);
  free(n);

  return Py_BuildValue("O", X_out);

}
