#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <math.h>

double HillFunction(double y, double m, double n, double h){

	double d,f;
	d = pow(y,n)/(pow(y,n) + pow(h,n));
	if (m > 0){
	    f = m*d;
	}
	else if(m < 0){
	    f = m*(1-d);
	}
	return f;
}

void HillSim(int N, int Nrecord, double T, int Nt, double* X, double* Y, 
	     double* x0, double* y0, double* noise_x, double* noise_y, double* mat, 
             double* a, double* b, double* c, double* bias, double* h,
             double* q, double* n){

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
      double dp = a[j]*X[(i-1)*N+j] - b[j]*Y[(i-1)*N+j];
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
        dr = dr + HillFunction(Y[(i-1)*N+k],mat[j*N+k],n[j*N+k],h[j*N+k]);
      }
      dr = dr - c[j]*X[(i-1)*N+j] + noise_x[i*N+j];
      X[i*N+j] = X[(i-1)*N+j] + dt*dr;

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
    PyObject* _x0 = PyList_GetItem(list, 4); //initial RNA concentrations
    PyObject* _y0 = PyList_GetItem(list, 5); //initial protein concentrations
    PyObject* _noise_x = PyList_GetItem(list, 6); //wiener process for RNA noise (in transcription and degradation)
    PyObject* _noise_y = PyList_GetItem(list, 7); //wiener process for protein noise (in translation and degradation)
    PyObject* _h = PyList_GetItem(list, 8); //concentration at half maximum
    PyObject* _mat = PyList_GetItem(list, 9); //maximum of hill function
    PyObject* _bias = PyList_GetItem(list, 10); //bias rate of transcription
    PyObject* _a = PyList_GetItem(list, 11); //translation rate
    PyObject* _b = PyList_GetItem(list, 12); //protein degradation rate
    PyObject* _c = PyList_GetItem(list, 13); //RNA degradation rate
    PyObject* _q = PyList_GetItem(list, 14); //noise amplitude
    PyObject* _n = PyList_GetItem(list, 15); //hill coefficient (power)

    double* X = malloc(N*Nt*sizeof(double));
    double* Y = malloc(N*Nt*sizeof(double));
    double* x0 = malloc(N*sizeof(double));
    double* y0 = malloc(N*sizeof(double));
    double* noise_x = malloc(N*Nt*sizeof(double));
    double* noise_y = malloc(N*Nt*sizeof(double));
    double* h = malloc(N*N*sizeof(double));
    double* mat = malloc(N*N*sizeof(double));
    double* bias = malloc(N*sizeof(double));
    double* a = malloc(N*sizeof(double));
    double* b = malloc(N*sizeof(double));
    double* c = malloc(N*sizeof(double));
    double* q = malloc(N*sizeof(double));
    double* n = malloc(N*N*sizeof(double));

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

    Py_ssize_t _h_size = PyList_Size(_h);
    for (Py_ssize_t j = 0; j < _h_size; j++) {
      h[j] = PyFloat_AsDouble(PyList_GetItem(_h, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _mat_size = PyList_Size(_mat);
    for (Py_ssize_t j = 0; j < _mat_size; j++) {
      mat[j] = PyFloat_AsDouble(PyList_GetItem(_mat, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _bias_size = PyList_Size(_bias);
    for (Py_ssize_t j = 0; j < _bias_size; j++) {
      bias[j] = PyFloat_AsDouble(PyList_GetItem(_bias, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _a_size = PyList_Size(_a);
    for (Py_ssize_t j = 0; j < _a_size; j++) {
      a[j] = PyFloat_AsDouble(PyList_GetItem(_a, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _b_size = PyList_Size(_b);
    for (Py_ssize_t j = 0; j < _b_size; j++) {
      b[j] = PyFloat_AsDouble(PyList_GetItem(_b, j));
      if (PyErr_Occurred()) return NULL;
    }

    Py_ssize_t _c_size = PyList_Size(_c);
    for (Py_ssize_t j = 0; j < _c_size; j++) {
      c[j] = PyFloat_AsDouble(PyList_GetItem(_c, j));
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

    HillSim(N, Nrecord, T, Nt, X, Y, x0, y0, noise_x, noise_y, mat, a, b, c, bias, h, q, n);

  npy_intp dims[2] = {Nt,N}; //row major order
  //Copy data into python list objects and free mem
  PyObject *X_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(X_out), X, N*Nt*sizeof(double));

  PyObject *Y_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA(Y_out), Y, N*Nt*sizeof(double));

  free(X);
  free(Y);
  free(x0);
  free(y0);
  free(noise_x);
  free(noise_y);
  free(h);
  free(mat);
  free(bias);
  free(a);
  free(b);
  free(c);
  free(q);
  free(n);

  return Py_BuildValue("(OO)", X_out, Y_out);

}
