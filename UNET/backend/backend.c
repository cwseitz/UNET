#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "hill.c"
#include "phixer.c"
#include "linear.c"


static PyMethodDef GRNMethods[] = {
    {"Hill", Hill, METH_VARARGS, "Python interface"},
    {"Linear", Linear, METH_VARARGS, "Python interface"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef backend = {
    PyModuleDef_HEAD_INIT,
    "backend",
    "Python interface for core implementations in C",
    -1,
    GRNMethods
};

PyMODINIT_FUNC PyInit_backend(void) {
    import_array();
    return PyModule_Create(&backend);
}
