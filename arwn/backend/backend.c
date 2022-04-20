#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "lmc_grn.c"
#include "lin_grn.c"


static PyMethodDef GRNMethods[] = {
    {"lmc_grn", lmc_grn, METH_VARARGS, "Python interface"},
    {"lin_grn", lin_grn, METH_VARARGS, "Python interface"},
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
