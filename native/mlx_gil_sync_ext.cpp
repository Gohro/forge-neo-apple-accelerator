// Exact-version MLX synchronization bridge for the bounded Phase 2C salvage.
//
// The sole blocking operation runs without the Python GIL.  No Python API,
// Python object access, or refcount operation is permitted between
// PyEval_SaveThread and PyEval_RestoreThread.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <exception>
#include <stdexcept>

#include <mlx/stream.h>
#include <mlx/version.h>

namespace {

PyObject* synchronize_without_gil(PyObject*, PyObject*) {
  std::exception_ptr failure;

  PyThreadState* thread_state = PyEval_SaveThread();
  try {
    mlx::core::synchronize();
  } catch (...) {
    failure = std::current_exception();
  }
  PyEval_RestoreThread(thread_state);

  if (failure) {
    try {
      std::rethrow_exception(failure);
    } catch (const std::exception& error) {
      PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
      PyErr_SetString(PyExc_RuntimeError, "unknown MLX synchronization failure");
    }
    return nullptr;
  }

  Py_RETURN_NONE;
}

PyObject* runtime_info(PyObject*, PyObject*) {
  return Py_BuildValue(
      "{s:s,s:i,s:i,s:i,s:s}",
      "mlx_version", mlx::core::version(),
      "mlx_version_major", MLX_VERSION_MAJOR,
      "mlx_version_minor", MLX_VERSION_MINOR,
      "mlx_version_patch", MLX_VERSION_PATCH,
      "gil_policy", "released_only_around_mlx_core_synchronize");
}

PyMethodDef methods[] = {
    {"synchronize", synchronize_without_gil, METH_NOARGS,
     "Synchronize the MLX default stream while the Python GIL is released."},
    {"runtime_info", runtime_info, METH_NOARGS,
     "Return the MLX headers/runtime identity compiled into this bridge."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "forge_apple_mlx_gil_sync_ext",
    "Exact-version MLX GIL-safe synchronization bridge.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_forge_apple_mlx_gil_sync_ext() {
  return PyModule_Create(&module);
}
