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

static struct PyModuleDef sentinel_core = {
    PyModuleDef_HEAD_INIT,
    "sentinel_core",
    "Python interface for core sentinel implementation in C",
    -1,
    GRNMethods
};

PyMODINIT_FUNC PyInit_sentinel_core(void) {
    import_array();
    return PyModule_Create(&sentinel_core);
}
