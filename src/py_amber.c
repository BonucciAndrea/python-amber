/* py_amber.c  -  CPython C extension binding for the Amber engine.
 *
 * Builds as `amber._amber` and links against libamber.so through the stable C
 * ABI declared in include/amber_capi.h. It does NOT link against, include, or
 * require NumPy, PyArrow or pandas -- not at build time and not at run time.
 *
 * That is a deliberate design decision, and it is the reason this file is
 * shaped the way it is. The obvious way to hand an Amber column to NumPy is to
 * #include <numpy/arrayobject.h> and call PyArray_SimpleNewFromData(). It also
 * pins the wheel to one NumPy ABI, makes NumPy a hard build dependency of an
 * engine binding that has nothing to do with NumPy, and gives PyArrow and
 * pandas nothing at all. So instead every Amber vector is exposed through
 * PEP 3118 -- the buffer protocol that is part of CPython itself -- and then
 *
 *     numpy.asarray(vec)          is zero-copy
 *     memoryview(vec)             is zero-copy
 *     pyarrow.py_buffer(vec)      is zero-copy
 *     torch.frombuffer(vec)       is zero-copy
 *
 * all fall out for free, from one implementation, with no third-party header
 * anywhere in the build.
 *
 * ---- lifetime ------------------------------------------------------------
 *
 * The one genuinely dangerous thing this module does is hand out interior
 * pointers into engine-owned memory. amber_get_vector_ptr()'s payload stays
 * valid exactly as long as someone holds a reference to the Amber value. The
 * buffer protocol is built for precisely this: bf_getbuffer() sets view->obj to
 * the owning Value and INCREFs it, so any consumer that keeps the memoryview
 * (NumPy stores it as the array's `base`) keeps the Amber refcount up, and the
 * payload cannot be freed underneath it. Value.__dealloc__ is the only place
 * that ever calls amber_release().
 *
 * ---- threads -------------------------------------------------------------
 *
 * The engine is single-threaded at its API boundary. Every call below runs with
 * the GIL HELD and never releases it, which makes the GIL the engine's lock for
 * free. This is why the module does not wrap long evaluations in
 * Py_BEGIN_ALLOW_THREADS: doing so would let a second Python thread re-enter
 * the interpreter mid-evaluation, and the resulting corruption would be
 * intermittent and unattributable. A caller that needs concurrency should run
 * several Amber PROCESSES, which is what the engine's own `peach` does.
 *
 * Amber - GNU AGPLv3 - see LICENSE and NOTICE.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "amber_capi.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* module state                                                              */
/* ------------------------------------------------------------------------- */

static PyObject *AmberError;       /* amber.AmberError  */
static int       amber_booted;

/* Raise AmberError carrying the engine's own error text. Always returns NULL so
 * call sites can `return amber_raise();`. */
static PyObject *amber_raise(void) {
    const char *msg = amber_last_error();
    PyErr_SetString(AmberError, (msg && *msg) ? msg : "amber: evaluation failed");
    return NULL;
}

/* ------------------------------------------------------------------------- */
/* Value: an owned handle on an Amber value                                  */
/* ------------------------------------------------------------------------- */

typedef struct {
    PyObject_HEAD
    amber_value handle;
    Py_ssize_t  exports;    /* live buffer exports; see Value_releasebuffer */
} ValueObject;

static PyTypeObject ValueType;

static PyObject *Value_wrap(amber_value handle) {
    ValueObject *self;
    if (!handle) return amber_raise();
    self = PyObject_New(ValueObject, &ValueType);
    if (!self) { amber_release(handle); return NULL; }
    self->handle = handle;
    self->exports = 0;
    return (PyObject *)self;
}

static void Value_dealloc(ValueObject *self) {
    if (self->handle) amber_release(self->handle);
    self->handle = 0;
    PyObject_Free(self);
}

/* PEP 3118 format code for an Amber element type, or NULL when the value has no
 * flat payload a consumer could safely walk.
 *
 * AMBER_T_BOOL is deliberately absent: it is genuinely bit-packed (one bit per
 * element, see the Tw[] table in the engine's a.h), and PEP 3118 has no
 * sub-byte format. Exposing it as bytes would silently give a consumer an array
 * eight times too short with the wrong values in it. It is converted instead. */
static const char *format_for(int type) {
    switch (type) {
        case AMBER_T_BYTE:  return "b";   /* int8   */
        case AMBER_T_SHORT: return "h";   /* int16  */
        case AMBER_T_INT:   return "i";   /* int32  */
        case AMBER_T_LONG:  return "q";   /* int64  */
        case AMBER_T_FLOAT: return "d";   /* float64*/
        case AMBER_T_CHAR:  return "B";   /* uint8; decode with .to_str()     */
        case AMBER_T_SYM:   return "i";   /* int32 symbol ids                 */
        default:            return NULL;
    }
}

static int Value_getbuffer(ValueObject *self, Py_buffer *view, int flags) {
    const void *data;
    const char *fmt;
    int type = 0, bits = 0;
    long long len = 0;

    if (!view) {
        PyErr_SetString(PyExc_BufferError, "NULL view in Value.__buffer__");
        return -1;
    }
    data = amber_get_vector_ptr(self->handle, &type, &len, &bits);
    fmt = format_for(type);
    if (!data || !fmt) {
        view->obj = NULL;
        PyErr_Format(PyExc_BufferError,
                     "amber.Value of type '%s' has no flat buffer to export "
                     "(use .items(), .to_str() or .scalar() instead)",
                     amber_type_name(self->handle));
        return -1;
    }

    view->buf = (void *)data;
    view->obj = (PyObject *)self;
    Py_INCREF(self);                     /* the whole point: pin the payload */
    view->len = (Py_ssize_t)len * (Py_ssize_t)(bits / 8);
    view->readonly = 1;                  /* engine memory is never ours to edit */
    view->itemsize = (Py_ssize_t)(bits / 8);
    view->format = (flags & PyBUF_FORMAT) ? (char *)fmt : NULL;
    view->ndim = 1;
    view->shape = NULL;
    view->strides = NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    if (flags & PyBUF_ND) {
        self->exports++;
        view->shape = PyMem_New(Py_ssize_t, 1);
        if (!view->shape) { Py_DECREF(self); self->exports--; return -1; }
        view->shape[0] = (Py_ssize_t)len;
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
            view->strides = PyMem_New(Py_ssize_t, 1);
            if (!view->strides) {
                PyMem_Free(view->shape);
                Py_DECREF(self); self->exports--;
                return -1;
            }
            view->strides[0] = view->itemsize;
        }
    }
    return 0;
}

static void Value_releasebuffer(ValueObject *self, Py_buffer *view) {
    if (view->shape)   { PyMem_Free(view->shape);   view->shape = NULL; }
    if (view->strides) { PyMem_Free(view->strides); view->strides = NULL; }
    if (self->exports > 0) self->exports--;
    /* view->obj is DECREF'd by CPython after this returns. */
}

static PyBufferProcs Value_as_buffer = {
    (getbufferproc)Value_getbuffer,
    (releasebufferproc)Value_releasebuffer
};

/* ---- Value methods ------------------------------------------------------- */

static PyObject *Value_type(ValueObject *self, PyObject *unused) {
    (void)unused; return PyLong_FromLong(amber_type(self->handle));
}
static PyObject *Value_type_name(ValueObject *self, PyObject *unused) {
    (void)unused; return PyUnicode_FromString(amber_type_name(self->handle));
}
static PyObject *Value_count(ValueObject *self, PyObject *unused) {
    (void)unused; return PyLong_FromLongLong(amber_count(self->handle));
}
static PyObject *Value_is_table(ValueObject *self, PyObject *unused) {
    (void)unused; return PyBool_FromLong(amber_is_table(self->handle));
}
static PyObject *Value_is_dict(ValueObject *self, PyObject *unused) {
    (void)unused; return PyBool_FromLong(amber_is_dict(self->handle));
}
static PyObject *Value_is_atom(ValueObject *self, PyObject *unused) {
    (void)unused; return PyBool_FromLong(amber_is_atom(self->handle));
}
static PyObject *Value_handle(ValueObject *self, PyObject *unused) {
    (void)unused; return PyLong_FromUnsignedLongLong(self->handle);
}

static PyObject *Value_flatten(ValueObject *self, PyObject *unused) {
    (void)unused; return Value_wrap(amber_flatten(self->handle));
}

static PyObject *Value_format(ValueObject *self, PyObject *args, PyObject *kw) {
    static char *kwlist[] = { "plain", NULL };
    int plain = 1;
    char *text;
    PyObject *res;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|p", kwlist, &plain)) return NULL;
    text = plain ? amber_format_plain(self->handle) : amber_format(self->handle);
    if (!text) return amber_raise();
    res = PyUnicode_DecodeUTF8(text, (Py_ssize_t)strlen(text), "replace");
    amber_free(text);
    return res;
}

static PyObject *Value_to_str(ValueObject *self, PyObject *unused) {
    long long need;
    char *buf;
    PyObject *res;
    (void)unused;
    need = amber_to_string(self->handle, NULL, 0);
    if (need < 0) {
        PyErr_Format(PyExc_TypeError, "amber.Value of type '%s' is not a string",
                     amber_type_name(self->handle));
        return NULL;
    }
    buf = (char *)PyMem_Malloc((size_t)need + 1);
    if (!buf) return PyErr_NoMemory();
    amber_to_string(self->handle, buf, need + 1);
    res = PyUnicode_DecodeUTF8(buf, (Py_ssize_t)need, "replace");
    PyMem_Free(buf);
    return res;
}

/* Decode a symbol vector (or symbol atom) to Python str objects. Symbols are
 * interned int32 ids in the payload; the text lives in the engine's symbol
 * table, which never moves and is never freed, so this is a pure copy-out. */
static PyObject *Value_to_symbols(ValueObject *self, PyObject *unused) {
    const void *data;
    int type = 0, bits = 0, ok = 0;
    long long len = 0, i;
    PyObject *list;
    (void)unused;

    if (amber_type(self->handle) == AMBER_T_SYM_ATOM) {
        int id = amber_to_symbol_id(self->handle, &ok);
        if (!ok) return amber_raise();
        return PyUnicode_FromString(amber_symbol_name(id));
    }
    data = amber_get_vector_ptr(self->handle, &type, &len, &bits);
    if (!data || type != AMBER_T_SYM) {
        PyErr_Format(PyExc_TypeError, "amber.Value of type '%s' is not a symbol vector",
                     amber_type_name(self->handle));
        return NULL;
    }
    list = PyList_New((Py_ssize_t)len);
    if (!list) return NULL;
    for (i = 0; i < len; i++) {
        PyObject *s = PyUnicode_FromString(amber_symbol_name(((const int *)data)[i]));
        if (!s) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, (Py_ssize_t)i, s);
    }
    return list;
}

static PyObject *Value_scalar(ValueObject *self, PyObject *unused) {
    int ok = 0, type = amber_type(self->handle);
    (void)unused;
    switch (type) {
        case AMBER_T_FLOAT_ATOM: {
            double d = amber_to_float(self->handle, &ok);
            if (!ok) break;
            return PyFloat_FromDouble(d);
        }
        case AMBER_T_INT_ATOM:
        case AMBER_T_LONG_ATOM:
        case AMBER_T_DATE:
        case AMBER_T_TIME:
        case AMBER_T_TIMESTAMP: {
            long long v = amber_to_int(self->handle, &ok);
            if (!ok) break;
            return PyLong_FromLongLong(v);
        }
        case AMBER_T_CHAR_ATOM:
            return Value_to_str(self, NULL);
        case AMBER_T_SYM_ATOM:
            return Value_to_symbols(self, NULL);
        default: break;
    }
    PyErr_Format(PyExc_TypeError, "amber.Value of type '%s' is not a scalar",
                 amber_type_name(self->handle));
    return NULL;
}

static PyObject *Value_ncols(ValueObject *self, PyObject *unused) {
    (void)unused; return PyLong_FromLong(amber_table_ncols(self->handle));
}

static PyObject *Value_colname(ValueObject *self, PyObject *arg) {
    long col = PyLong_AsLong(arg);
    const char *nm;
    if (col == -1 && PyErr_Occurred()) return NULL;
    nm = amber_table_colname(self->handle, (int)col);
    if (!nm) { PyErr_SetString(PyExc_IndexError, "column index out of range"); return NULL; }
    return PyUnicode_FromString(nm);
}

static PyObject *Value_column(ValueObject *self, PyObject *arg) {
    long col = PyLong_AsLong(arg);
    amber_value c;
    if (col == -1 && PyErr_Occurred()) return NULL;
    c = amber_table_column(self->handle, (int)col);
    if (!c) { PyErr_SetString(PyExc_IndexError, "column index out of range"); return NULL; }
    return Value_wrap(c);
}

static PyObject *Value_keys(ValueObject *self, PyObject *unused) {
    (void)unused; return Value_wrap(amber_dict_keys(self->handle));
}
static PyObject *Value_values(ValueObject *self, PyObject *unused) {
    (void)unused; return Value_wrap(amber_dict_values(self->handle));
}

/* Explode a general list (`A) into a Python list of Values. Used for dict
 * values, nested lists and anything else with no flat payload. */
static PyObject *Value_items(ValueObject *self, PyObject *unused) {
    long long n = amber_count(self->handle), i;
    PyObject *list;
    (void)unused;
    if (n < 0) { PyErr_SetString(AmberError, "value has no item count"); return NULL; }
    list = PyList_New((Py_ssize_t)n);
    if (!list) return NULL;
    for (i = 0; i < n; i++) {
        /* `x@i` -- indexing is the engine's own accessor and works uniformly for
         * lists, dict values and nested structures. Building it as source once
         * per element would be absurd, so it goes through the engine's index
         * verb via a cached global helper installed at init time. */
        amber_value idx = amber_int(i);
        amber_value args[2];
        amber_value item;
        PyObject *wrapped;
        args[0] = self->handle;
        args[1] = idx;
        item = amber_call("pyamber.at", args, 2);
        amber_release(idx);
        if (!item) { Py_DECREF(list); return amber_raise(); }
        wrapped = Value_wrap(item);
        if (!wrapped) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, (Py_ssize_t)i, wrapped);
    }
    return list;
}

static Py_ssize_t Value_length(ValueObject *self) {
    long long n = amber_count(self->handle);
    return n < 0 ? 0 : (Py_ssize_t)n;
}

static PyObject *Value_repr(ValueObject *self) {
    char *text = amber_format_plain(self->handle);
    PyObject *res;
    if (!text) return PyUnicode_FromFormat("<amber.Value %s>", amber_type_name(self->handle));
    res = PyUnicode_FromFormat("<amber.Value %s>\n%s", amber_type_name(self->handle), text);
    amber_free(text);
    return res;
}

static PyMethodDef Value_methods[] = {
    { "type",       (PyCFunction)Value_type,       METH_NOARGS,  "Amber type code (see amber.T_*)." },
    { "type_name",  (PyCFunction)Value_type_name,  METH_NOARGS,  "Human-readable type name." },
    { "count",      (PyCFunction)Value_count,      METH_NOARGS,  "Element count (rows, for a table)." },
    { "is_table",   (PyCFunction)Value_is_table,   METH_NOARGS,  "True if this is an Amber table." },
    { "is_dict",    (PyCFunction)Value_is_dict,    METH_NOARGS,  "True if this is an Amber dictionary." },
    { "is_atom",    (PyCFunction)Value_is_atom,    METH_NOARGS,  "True if this is a scalar atom." },
    { "handle",     (PyCFunction)Value_handle,     METH_NOARGS,  "The raw 64-bit engine handle (diagnostics only)." },
    { "flatten",    (PyCFunction)Value_flatten,    METH_NOARGS,  "Materialise a lazy range; identity otherwise." },
    { "format",     (PyCFunction)Value_format,     METH_VARARGS | METH_KEYWORDS, "Render as the REPL would. format(plain=True) strips ANSI colour." },
    { "to_str",     (PyCFunction)Value_to_str,     METH_NOARGS,  "Decode a char vector to str." },
    { "to_symbols", (PyCFunction)Value_to_symbols, METH_NOARGS,  "Decode a symbol vector to a list of str." },
    { "scalar",     (PyCFunction)Value_scalar,     METH_NOARGS,  "Convert an atom to the natural Python scalar." },
    { "ncols",      (PyCFunction)Value_ncols,      METH_NOARGS,  "Column count of a table, or -1." },
    { "colname",    (PyCFunction)Value_colname,    METH_O,       "Name of column i." },
    { "column",     (PyCFunction)Value_column,     METH_O,       "Column i as a Value (flattened)." },
    { "keys",       (PyCFunction)Value_keys,       METH_NOARGS,  "Dictionary keys as a Value." },
    { "values",     (PyCFunction)Value_values,     METH_NOARGS,  "Dictionary values as a Value." },
    { "items",      (PyCFunction)Value_items,      METH_NOARGS,  "Explode a general list into a list of Values." },
    { NULL, NULL, 0, NULL }
};

static PySequenceMethods Value_as_sequence = { (lenfunc)Value_length, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static PyTypeObject ValueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "amber._amber.Value",
    .tp_basicsize = sizeof(ValueObject),
    .tp_dealloc   = (destructor)Value_dealloc,
    .tp_repr      = (reprfunc)Value_repr,
    .tp_as_sequence = &Value_as_sequence,
    .tp_as_buffer = &Value_as_buffer,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "An owned handle on a value inside the Amber engine.\n\n"
                    "Flat vectors export the PEP 3118 buffer protocol, so\n"
                    "numpy.asarray(v), memoryview(v) and pyarrow.py_buffer(v)\n"
                    "are all zero-copy and all keep the engine value alive for\n"
                    "as long as they exist.",
    .tp_methods   = Value_methods,
};

/* ------------------------------------------------------------------------- */
/* Python -> Amber conversion                                                */
/* ------------------------------------------------------------------------- */

static amber_value to_amber(PyObject *obj);   /* forward */

/* Convert a PEP 3118 buffer (a NumPy array, an array.array, another Value, a
 * memoryview) into an Amber vector. This is a COPY: Amber stores its own
 * 32-byte object header immediately before every payload, so it cannot adopt a
 * foreign allocation as a native vector. (The reverse direction -- Amber to
 * NumPy -- IS zero-copy, which is the direction that carries the volume.) */
static amber_value buffer_to_amber(PyObject *obj) {
    Py_buffer view;
    amber_value res = 0;
    Py_ssize_t i, n;

    if (PyObject_GetBuffer(obj, &view, PyBUF_FORMAT | PyBUF_ANY_CONTIGUOUS) != 0) {
        PyErr_Clear();
        return 0;
    }
    if (view.ndim > 1) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_TypeError,
                        "amber: only 1-D buffers convert to an Amber vector "
                        "(ravel() or pass a list of vectors)");
        return 0;
    }
    n = view.itemsize ? view.len / view.itemsize : 0;

    if (!view.format) {                    /* untyped bytes */
        res = amber_from_bytes((const char *)view.buf, (long long)view.len);
    } else {
        char code = view.format[0];
        if (code == '@' || code == '=' || code == '<' || code == '>' || code == '!') code = view.format[1];
        switch (code) {
            case 'd': res = amber_from_float64((const double *)view.buf, (long long)n); break;
            case 'q': case 'Q':
                res = amber_from_int64((const long long *)view.buf, (long long)n); break;
            case 'l': case 'L':
                res = (sizeof(long) == 8)
                    ? amber_from_int64((const long long *)view.buf, (long long)n)
                    : amber_from_int32((const int *)view.buf, (long long)n);
                break;
            case 'i': case 'I': res = amber_from_int32((const int *)view.buf, (long long)n); break;
            case 'c': case 'B': case 's':
                res = amber_from_bytes((const char *)view.buf, (long long)view.len); break;
            case 'f': {                    /* float32 -> widen to Amber's float64 */
                double *tmp = (double *)PyMem_Malloc((size_t)n * sizeof(double) + 1);
                if (!tmp) { PyBuffer_Release(&view); PyErr_NoMemory(); return 0; }
                for (i = 0; i < n; i++) tmp[i] = (double)((const float *)view.buf)[i];
                res = amber_from_float64(tmp, (long long)n);
                PyMem_Free(tmp);
                break;
            }
            case 'b': case 'h': case 'H': { /* narrow ints -> widen to int64 */
                long long *tmp = (long long *)PyMem_Malloc((size_t)n * sizeof(long long) + 1);
                if (!tmp) { PyBuffer_Release(&view); PyErr_NoMemory(); return 0; }
                for (i = 0; i < n; i++)
                    tmp[i] = (code == 'b') ? (long long)((const signed char *)view.buf)[i]
                           : (code == 'h') ? (long long)((const short *)view.buf)[i]
                                           : (long long)((const unsigned short *)view.buf)[i];
                res = amber_from_int64(tmp, (long long)n);
                PyMem_Free(tmp);
                break;
            }
            case '?': {                     /* bool -> 0/1 int64 */
                long long *tmp = (long long *)PyMem_Malloc((size_t)n * sizeof(long long) + 1);
                if (!tmp) { PyBuffer_Release(&view); PyErr_NoMemory(); return 0; }
                for (i = 0; i < n; i++) tmp[i] = ((const unsigned char *)view.buf)[i] ? 1 : 0;
                res = amber_from_int64(tmp, (long long)n);
                PyMem_Free(tmp);
                break;
            }
            default:
                PyBuffer_Release(&view);
                PyErr_Format(PyExc_TypeError,
                             "amber: buffer format '%s' has no Amber equivalent", view.format);
                return 0;
        }
    }
    PyBuffer_Release(&view);
    return res;
}

static amber_value sequence_to_amber(PyObject *obj) {
    Py_ssize_t n = PySequence_Size(obj), i;
    amber_value *slot, res;
    if (n < 0) return 0;
    slot = (amber_value *)PyMem_Calloc((size_t)(n ? n : 1), sizeof(amber_value));
    if (!slot) { PyErr_NoMemory(); return 0; }
    for (i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) goto fail;
        slot[i] = to_amber(item);
        Py_DECREF(item);
        if (!slot[i]) goto fail;
    }
    res = amber_make_list(slot, (long long)n);
    for (i = 0; i < n; i++) amber_release(slot[i]);
    PyMem_Free(slot);
    if (!res) return (amber_value)(amber_raise() ? 0 : 0);
    return res;
fail:
    for (i = 0; i < n; i++) if (slot[i]) amber_release(slot[i]);
    PyMem_Free(slot);
    return 0;
}

static amber_value mapping_to_amber(PyObject *obj) {
    PyObject *keys = PyMapping_Keys(obj);
    Py_ssize_t n, i;
    const char **names = NULL;
    amber_value *vals = NULL, res = 0;
    PyObject **owned = NULL;

    if (!keys) return 0;
    n = PyList_Size(keys);
    names = (const char **)PyMem_Calloc((size_t)(n ? n : 1), sizeof(char *));
    vals  = (amber_value *)PyMem_Calloc((size_t)(n ? n : 1), sizeof(amber_value));
    owned = (PyObject **)PyMem_Calloc((size_t)(n ? n : 1), sizeof(PyObject *));
    if (!names || !vals || !owned) { PyErr_NoMemory(); goto done; }

    for (i = 0; i < n; i++) {
        PyObject *k = PyList_GetItem(keys, i);            /* borrowed */
        PyObject *v = PyObject_GetItem(obj, k);
        PyObject *ks;
        if (!v) goto done;
        ks = PyObject_Str(k);
        if (!ks) { Py_DECREF(v); goto done; }
        owned[i] = ks;                                     /* keeps the UTF-8 alive */
        names[i] = PyUnicode_AsUTF8(ks);
        vals[i] = to_amber(v);
        Py_DECREF(v);
        if (!names[i] || !vals[i]) goto done;
    }
    res = amber_make_dict(names, vals, (int)n);
done:
    if (vals)  { for (i = 0; i < n; i++) if (vals[i]) amber_release(vals[i]); PyMem_Free(vals); }
    if (owned) { for (i = 0; i < n; i++) Py_XDECREF(owned[i]); PyMem_Free(owned); }
    PyMem_Free((void *)names);
    Py_DECREF(keys);
    return res;
}

static amber_value to_amber(PyObject *obj) {
    if (obj == Py_None) return amber_null();
    if (PyObject_TypeCheck(obj, &ValueType)) return amber_retain(((ValueObject *)obj)->handle);
    if (PyBool_Check(obj)) return amber_int(obj == Py_True ? 1 : 0);
    if (PyLong_Check(obj)) {
        long long v = PyLong_AsLongLong(obj);
        if (v == -1 && PyErr_Occurred()) return 0;
        return amber_int(v);
    }
    if (PyFloat_Check(obj)) return amber_float(PyFloat_AsDouble(obj));
    if (PyUnicode_Check(obj)) {
        Py_ssize_t n = 0;
        const char *s = PyUnicode_AsUTF8AndSize(obj, &n);
        if (!s) return 0;
        return amber_from_bytes(s, (long long)n);
    }
    if (PyBytes_Check(obj))
        return amber_from_bytes(PyBytes_AS_STRING(obj), (long long)PyBytes_GET_SIZE(obj));
    if (PyDict_Check(obj) || PyMapping_Check(obj)) {
        if (!PySequence_Check(obj)) return mapping_to_amber(obj);
    }
    if (PyObject_CheckBuffer(obj)) {
        amber_value res = buffer_to_amber(obj);
        if (res || PyErr_Occurred()) return res;
    }
    if (PySequence_Check(obj)) return sequence_to_amber(obj);
    PyErr_Format(PyExc_TypeError, "amber: cannot convert %s to an Amber value",
                 Py_TYPE(obj)->tp_name);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* module functions                                                          */
/* ------------------------------------------------------------------------- */

/* Helpers the binding installs once, so hot paths go through amber_call()
 * rather than through re-parsing a source string per element. */
/* Helpers the binding installs once, so hot paths go through amber_call() with
 * native arguments rather than through formatting and re-parsing a source
 * string per element.
 *
 * They live in the `pyamber` NAMESPACE rather than being called pyamber_at and
 * friends, and that is not a style choice: `_` is a VERB in Amber (drop, and
 * floor), so `pyamber_at` does not lex as one identifier at all -- it lexes as
 * `pyamber _ at` and fails with an undefined-name error on a name the caller
 * never wrote. Dotted names are the language's own namespacing and lex
 * correctly. */
static const char *BOOTSTRAP =
    "pyamber.at:{x@y}\n"
    "pyamber.sym:{`$x}\n"
    "pyamber.widen:{0+x}\n"
    /* Flatten a keyed table into an ordinary one, key columns first. Written
     * out of core primitives rather than calling amber.k's `unkey` so that it
     * works on a BARE engine too -- a satellite that boots without the standard
     * library still gets sane `select ... by ...` results. Verified identical
     * to `unkey` (`(unkey x)~pyamber.unkey x` is 1). */
    "pyamber.unkey:{k:+!x;v:+.x;+((!k),(!v))!((.k),(.v))}\n"
    "pyamber.keycols:{!+!x}\n";

static PyObject *py_init(PyObject *self, PyObject *args) {
    const char *home = NULL;
    amber_value res;
    (void)self;
    if (!PyArg_ParseTuple(args, "|z", &home)) return NULL;
    if (amber_abi_version() != AMBER_CAPI_ABI) {
        PyErr_Format(PyExc_ImportError,
                     "amber: libamber ABI mismatch -- this build of python-amber "
                     "was compiled against ABI %d but the loaded libamber reports "
                     "ABI %d. Rebuild one of them.",
                     AMBER_CAPI_ABI, amber_abi_version());
        return NULL;
    }
    if (amber_init(home) != 0) return amber_raise();
    amber_set_diagnostics(0);      /* errors become exceptions, not stderr noise */
    if (!amber_booted) {
        res = amber_eval_str(BOOTSTRAP);
        if (!res) return amber_raise();
        amber_release(res);
        amber_booted = 1;
    }
    Py_RETURN_NONE;
}

static PyObject *py_shutdown(PyObject *self, PyObject *unused) {
    (void)self; (void)unused; amber_shutdown(); Py_RETURN_NONE;
}
static PyObject *py_version(PyObject *self, PyObject *unused) {
    (void)self; (void)unused; return PyUnicode_FromString(amber_version_string());
}
static PyObject *py_abi(PyObject *self, PyObject *unused) {
    (void)self; (void)unused; return PyLong_FromLong(amber_abi_version());
}
static PyObject *py_set_diagnostics(PyObject *self, PyObject *arg) {
    int on = PyObject_IsTrue(arg);
    (void)self;
    if (on < 0) return NULL;
    return PyLong_FromLong(amber_set_diagnostics(on));
}

static PyObject *py_eval(PyObject *self, PyObject *arg) {
    const char *src = PyUnicode_AsUTF8(arg);
    (void)self;
    if (!src) return NULL;
    return Value_wrap(amber_eval_str(src));
}

static PyObject *py_eval_qsql(PyObject *self, PyObject *arg) {
    const char *src = PyUnicode_AsUTF8(arg);
    (void)self;
    if (!src) return NULL;
    return Value_wrap(amber_eval_qsql(src));
}

static PyObject *py_call(PyObject *self, PyObject *args) {
    const char *name;
    PyObject *argseq;
    Py_ssize_t n, i;
    amber_value slot[8], res;
    (void)self;

    if (!PyArg_ParseTuple(args, "sO", &name, &argseq)) return NULL;
    if (!PySequence_Check(argseq)) {
        PyErr_SetString(PyExc_TypeError, "amber.call: second argument must be a sequence");
        return NULL;
    }
    n = PySequence_Size(argseq);
    if (n < 0) return NULL;
    if (n > 8) {
        PyErr_SetString(PyExc_TypeError,
                        "amber.call: the engine applies at most 8 arguments; "
                        "pass a list as one argument instead");
        return NULL;
    }
    for (i = 0; i < n; i++) slot[i] = 0;
    for (i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(argseq, i);
        if (!item) goto fail;
        slot[i] = to_amber(item);
        Py_DECREF(item);
        if (!slot[i]) goto fail;
    }
    res = amber_call(name, slot, (int)n);
    for (i = 0; i < n; i++) amber_release(slot[i]);
    return Value_wrap(res);
fail:
    for (i = 0; i < n; i++) if (slot[i]) amber_release(slot[i]);
    return NULL;
}

static PyObject *py_convert(PyObject *self, PyObject *arg) {
    amber_value v = to_amber(arg);
    (void)self;
    if (!v) { if (!PyErr_Occurred()) amber_raise(); return NULL; }
    return Value_wrap(v);
}

static PyObject *py_set_global(PyObject *self, PyObject *args) {
    const char *name;
    PyObject *obj;
    amber_value v;
    int rc;
    (void)self;
    if (!PyArg_ParseTuple(args, "sO", &name, &obj)) return NULL;
    v = to_amber(obj);
    if (!v) { if (!PyErr_Occurred()) amber_raise(); return NULL; }
    rc = amber_set_global(name, v);
    amber_release(v);
    if (rc != 0) return amber_raise();
    Py_RETURN_NONE;
}

static PyObject *py_get_global(PyObject *self, PyObject *arg) {
    const char *name = PyUnicode_AsUTF8(arg);
    (void)self;
    if (!name) return NULL;
    return Value_wrap(amber_get_global(name));
}

static PyObject *py_make_table(PyObject *self, PyObject *args) {
    PyObject *names_seq, *cols_seq;
    Py_ssize_t n, i;
    const char **names = NULL;
    amber_value *cols = NULL, res;
    PyObject **owned = NULL;
    PyObject *out = NULL;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &names_seq, &cols_seq)) return NULL;
    n = PySequence_Size(names_seq);
    if (n < 0 || PySequence_Size(cols_seq) != n) {
        PyErr_SetString(PyExc_ValueError, "amber.make_table: names and columns must be the same length");
        return NULL;
    }
    names = (const char **)PyMem_Calloc((size_t)(n ? n : 1), sizeof(char *));
    cols  = (amber_value *)PyMem_Calloc((size_t)(n ? n : 1), sizeof(amber_value));
    owned = (PyObject **)PyMem_Calloc((size_t)(n ? n : 1), sizeof(PyObject *));
    if (!names || !cols || !owned) { PyErr_NoMemory(); goto done; }

    for (i = 0; i < n; i++) {
        PyObject *nm = PySequence_GetItem(names_seq, i);
        PyObject *cv = PySequence_GetItem(cols_seq, i);
        PyObject *ns;
        if (!nm || !cv) { Py_XDECREF(nm); Py_XDECREF(cv); goto done; }
        ns = PyObject_Str(nm);
        Py_DECREF(nm);
        if (!ns) { Py_DECREF(cv); goto done; }
        owned[i] = ns;
        names[i] = PyUnicode_AsUTF8(ns);
        cols[i] = to_amber(cv);
        Py_DECREF(cv);
        if (!names[i] || !cols[i]) goto done;
    }
    res = amber_make_table(names, cols, (int)n);
    out = Value_wrap(res);
done:
    if (cols)  { for (i = 0; i < n; i++) if (cols[i]) amber_release(cols[i]); PyMem_Free(cols); }
    if (owned) { for (i = 0; i < n; i++) Py_XDECREF(owned[i]); PyMem_Free(owned); }
    PyMem_Free((void *)names);
    return out;
}

/* Arrow: the addresses cross as plain Python ints, exactly as PyArrow's own
 * _import_from_c/_export_to_c expect. Nothing here knows what an ArrowSchema
 * looks like, which is why this module has no Arrow build dependency.
 *
 * py_arrow_export() hands back two heap structs the CALLER must free with
 * free_ptr() -- an Arrow "move" transfers contents, never containers, so
 * importing from them leaves the containers behind. That is 152 bytes per
 * export: invisible in a notebook and fatal in a Flight server that exports a
 * table per request. amber/table.py therefore uses arrow_export_into(), which
 * moves into structs Python already owns and leaves nothing to free; this
 * lower-level entry point is kept for callers that need the raw addresses. */
static PyObject *py_arrow_export(PyObject *self, PyObject *arg) {
    void *schema = NULL, *array = NULL;
    (void)self;
    if (!PyObject_TypeCheck(arg, &ValueType)) {
        PyErr_SetString(PyExc_TypeError, "amber.arrow_export: expected an amber Value");
        return NULL;
    }
    if (amber_arrow_export(((ValueObject *)arg)->handle, &schema, &array) != 0)
        return amber_raise();
    return Py_BuildValue("(KK)", (unsigned long long)(size_t)schema,
                                 (unsigned long long)(size_t)array);
}

/* The form a binding actually wants: the caller allocates the two Arrow structs
 * (here, ctypes objects owned by Python -- see amber/arrow_abi.py) and the
 * engine MOVES its export into them. Nothing is left for anyone to free, and no
 * pointer allocated by one allocator is ever passed to another's free(). */
static PyObject *py_arrow_export_into(PyObject *self, PyObject *args) {
    PyObject *value;
    unsigned long long schema, array;
    (void)self;
    if (!PyArg_ParseTuple(args, "OKK", &value, &schema, &array)) return NULL;
    if (!PyObject_TypeCheck(value, &ValueType)) {
        PyErr_SetString(PyExc_TypeError, "amber.arrow_export_into: expected an amber Value");
        return NULL;
    }
    if (!schema || !array) {
        PyErr_SetString(PyExc_ValueError, "amber.arrow_export_into: null struct address");
        return NULL;
    }
    if (amber_arrow_export_into(((ValueObject *)value)->handle,
                                (void *)(size_t)schema, (void *)(size_t)array) != 0)
        return amber_raise();
    Py_RETURN_NONE;
}

static PyObject *py_arrow_import(PyObject *self, PyObject *args) {
    unsigned long long schema, array;
    (void)self;
    if (!PyArg_ParseTuple(args, "KK", &schema, &array)) return NULL;
    return Value_wrap(amber_arrow_import((void *)(size_t)schema, (void *)(size_t)array));
}

static PyObject *py_plugin_load(PyObject *self, PyObject *arg) {
    const char *path = PyUnicode_AsUTF8(arg);
    (void)self;
    if (!path) return NULL;
    if (amber_plugin_load(path) != 0) return amber_raise();
    Py_RETURN_NONE;
}

/* free() an address the engine handed out. Needed for exactly one thing: after
 * PyArrow's Table._import_from_c() has MOVED the contents out of an ArrowSchema
 * / ArrowArray pair, the two heap structs those contents lived in are still the
 * caller's to release -- the Arrow C data interface transfers the contents, not
 * the containers. Freeing them from Python via ctypes would work and would also
 * be a way to hand an arbitrary integer to free(); routing it through the
 * engine's own allocator entry point keeps that one dangerous operation inside
 * the binding, where its single legitimate caller is visible. */
static PyObject *py_free_ptr(PyObject *self, PyObject *arg) {
    unsigned long long addr = PyLong_AsUnsignedLongLong(arg);
    (void)self;
    if (PyErr_Occurred()) return NULL;
    if (addr) amber_free((void *)(size_t)addr);
    Py_RETURN_NONE;
}

/* Build a symbol vector from an iterable of strings.
 *
 * Symbols are Amber's interned-string type and the idiomatic representation for
 * a low-cardinality text column -- a ticker, a venue, an exchange code. They are
 * stored as int32 ids into a process-wide table, so a symbol column costs four
 * bytes per row regardless of how long the text is, and comparing two of them is
 * comparing two integers. That is why `sym` columns are what every Amber query
 * groups and joins on, and why a pandas object/string column should become one
 * rather than a list of char vectors. */
static PyObject *py_symbols(PyObject *self, PyObject *arg) {
    PyObject *seq = PySequence_Fast(arg, "amber.symbols: expected an iterable of str");
    Py_ssize_t n, i;
    const char **names = NULL;
    PyObject **owned = NULL;
    PyObject *out = NULL;
    (void)self;

    if (!seq) return NULL;
    n = PySequence_Fast_GET_SIZE(seq);
    names = (const char **)PyMem_Calloc((size_t)(n ? n : 1), sizeof(char *));
    owned = (PyObject **)PyMem_Calloc((size_t)(n ? n : 1), sizeof(PyObject *));
    if (!names || !owned) { PyErr_NoMemory(); goto done; }
    for (i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);   /* borrowed */
        PyObject *text = PyObject_Str(item);
        if (!text) goto done;
        owned[i] = text;
        names[i] = PyUnicode_AsUTF8(text);
        if (!names[i]) goto done;
    }
    out = Value_wrap(amber_from_symbols(names, (long long)n));
done:
    if (owned) { for (i = 0; i < n; i++) Py_XDECREF(owned[i]); PyMem_Free(owned); }
    PyMem_Free((void *)names);
    Py_DECREF(seq);
    return out;
}

static PyObject *py_symbol_name(PyObject *self, PyObject *arg) {
    long id = PyLong_AsLong(arg);
    (void)self;
    if (id == -1 && PyErr_Occurred()) return NULL;
    return PyUnicode_FromString(amber_symbol_name((int)id));
}

static PyMethodDef module_methods[] = {
    { "init",            py_init,            METH_VARARGS, "init(home=None) -- boot the engine, optionally loading the .k standard library from `home`." },
    { "shutdown",        py_shutdown,        METH_NOARGS,  "Release the engine's scratch arena." },
    { "version",         py_version,         METH_NOARGS,  "Engine version string." },
    { "abi",             py_abi,             METH_NOARGS,  "Engine C-API ABI number." },
    { "set_diagnostics", py_set_diagnostics, METH_O,       "Enable/disable the engine's rich stderr diagnostics." },
    { "eval",            py_eval,            METH_O,       "eval(src) -> Value.  Evaluate Amber source; the last statement's value." },
    { "eval_qsql",       py_eval_qsql,       METH_O,       "eval_qsql(line) -> Value.  As eval(), after the bare-qSQL rewrite." },
    { "call",            py_call,            METH_VARARGS, "call(name, args) -> Value.  Apply a global function to 0..8 arguments." },
    { "convert",         py_convert,         METH_O,       "convert(obj) -> Value.  Python object to Amber value." },
    { "set_global",      py_set_global,      METH_VARARGS, "set_global(name, obj) -- bind an Amber global." },
    { "get_global",      py_get_global,      METH_O,       "get_global(name) -> Value." },
    { "make_table",      py_make_table,      METH_VARARGS, "make_table(names, columns) -> Value." },
    { "arrow_export",    py_arrow_export,    METH_O,       "arrow_export(table) -> (schema_addr, array_addr).  The two STRUCTS become yours: call free_ptr() on both once their contents have been imported or released.  Prefer arrow_export_into(), which leaves nothing to free." },
    { "arrow_export_into", py_arrow_export_into, METH_VARARGS, "arrow_export_into(table, schema_addr, array_addr) -- move the export into caller-owned structs." },
    { "arrow_import",    py_arrow_import,    METH_VARARGS, "arrow_import(schema_addr, array_addr) -> Value.  Moves the two structs." },
    { "plugin_load",     py_plugin_load,     METH_O,       "plugin_load(path) -- dlopen a native Amber plugin." },
    { "symbols",         py_symbols,         METH_O,       "symbols(iterable_of_str) -> Value.  An Amber symbol vector." },
    { "symbol_name",     py_symbol_name,     METH_O,       "symbol_name(id) -> str." },
    { "free_ptr",        py_free_ptr,        METH_O,       "free_ptr(addr) -- release an engine-allocated struct after Arrow moved its contents out." },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef amber_module = {
    PyModuleDef_HEAD_INIT,
    "amber._amber",
    "Low-level CPython binding for the Amber columnar time-series engine.\n"
    "Links libamber.so through its stable C ABI. Use the `amber` package.",
    -1,
    module_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__amber(void) {
    PyObject *mod;
    if (PyType_Ready(&ValueType) < 0) return NULL;
    mod = PyModule_Create(&amber_module);
    if (!mod) return NULL;

    AmberError = PyErr_NewExceptionWithDoc(
        "amber.AmberError",
        "Raised when the Amber engine reports an error.\n\n"
        "The message is the engine's own diagnostic text -- the same string an\n"
        "Amber-level .[f;args;handler] trap would have received.",
        NULL, NULL);
    if (!AmberError) { Py_DECREF(mod); return NULL; }
    Py_INCREF(AmberError);
    if (PyModule_AddObject(mod, "AmberError", AmberError) < 0) {
        Py_DECREF(AmberError); Py_DECREF(AmberError); Py_DECREF(mod); return NULL;
    }
    Py_INCREF(&ValueType);
    if (PyModule_AddObject(mod, "Value", (PyObject *)&ValueType) < 0) {
        Py_DECREF(&ValueType); Py_DECREF(mod); return NULL;
    }

    /* The generic null `::` as a raw handle. A binding needs to recognise it
     * because it is what an ASSIGNMENT and any other statement-with-no-value
     * evaluates to, and displaying "<amber.Value function>" after every `t:1`
     * is not what anyone means by "the value of the cell". Exported as the
     * handle rather than as a singleton object so the comparison is an integer
     * compare with no allocation. */
    if (PyModule_AddObject(mod, "NULL_HANDLE",
                           PyLong_FromUnsignedLongLong(amber_null())) < 0) {
        Py_DECREF(mod); return NULL;
    }

#define ADDT(name) if (PyModule_AddIntConstant(mod, #name, name) < 0) { Py_DECREF(mod); return NULL; }
    ADDT(AMBER_T_LIST)  ADDT(AMBER_T_RANGE) ADDT(AMBER_T_BOOL)  ADDT(AMBER_T_BYTE)
    ADDT(AMBER_T_SHORT) ADDT(AMBER_T_INT)   ADDT(AMBER_T_LONG)  ADDT(AMBER_T_FLOAT)
    ADDT(AMBER_T_CHAR)  ADDT(AMBER_T_SYM)   ADDT(AMBER_T_TABLE) ADDT(AMBER_T_DICT)
    ADDT(AMBER_T_INT_ATOM) ADDT(AMBER_T_LONG_ATOM) ADDT(AMBER_T_FLOAT_ATOM)
    ADDT(AMBER_T_CHAR_ATOM) ADDT(AMBER_T_SYM_ATOM)
    ADDT(AMBER_T_DATE)  ADDT(AMBER_T_TIME)  ADDT(AMBER_T_TIMESTAMP)
    ADDT(AMBER_CAPI_ABI)
#undef ADDT
    return mod;
}
