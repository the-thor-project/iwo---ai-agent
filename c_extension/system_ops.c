/*
 * System Operations C Extension Module
 * Provides optimized system operations for the chatbot
 */

#include <Python.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

/* Get current timestamp */
static PyObject* get_timestamp(PyObject* self, PyObject* args) {
    time_t now = time(NULL);
    struct tm* timeinfo = localtime(&now);
    char buffer[100];
    
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    return PyUnicode_FromString(buffer);
}

/* Fast string hashing for embeddings */
static PyObject* fast_hash(PyObject* self, PyObject* args) {
    const char* text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }
    
    unsigned long hash = 5381;
    int c;
    
    while ((c = *text++)) {
        hash = ((hash << 5) + hash) + c;
    }
    
    return PyLong_FromLong((long)(hash % 1000000007));
}

/* Calculate text complexity */
static PyObject* text_complexity(PyObject* self, PyObject* args) {
    const char* text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }
    
    int words = 0, syllables = 0;
    int in_word = 0;
    
    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] == ' ' || text[i] == '\n' || text[i] == '\t') {
            if (in_word) {
                words++;
                in_word = 0;
            }
        } else {
            in_word = 1;
            // Simple syllable counting
            if (text[i] == 'a' || text[i] == 'e' || text[i] == 'i' || 
                text[i] == 'o' || text[i] == 'u') {
                if (i == 0 || (text[i-1] != 'a' && text[i-1] != 'e' && 
                    text[i-1] != 'i' && text[i-1] != 'o' && text[i-1] != 'u')) {
                    syllables++;
                }
            }
        }
    }
    
    if (in_word) words++;
    syllables = syllables > 0 ? syllables : 1;
    
    // Flesch Kincaid Grade
    float grade = 0.39 * (float)words + 11.8 * (float)syllables / (float)words - 15.59;
    
    return PyFloat_FromDouble(grade > 0 ? grade : 0);
}

/* Optimize text by removing extra spaces */
static PyObject* optimize_text(PyObject* self, PyObject* args) {
    const char* text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }
    
    char* buffer = (char*)malloc(strlen(text) + 1);
    int buffer_pos = 0;
    int prev_space = 0;
    
    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] == ' ' || text[i] == '\n' || text[i] == '\t') {
            if (!prev_space && buffer_pos > 0) {
                buffer[buffer_pos++] = ' ';
                prev_space = 1;
            }
        } else {
            buffer[buffer_pos++] = text[i];
            prev_space = 0;
        }
    }
    
    buffer[buffer_pos] = '\0';
    PyObject* result = PyUnicode_FromString(buffer);
    free(buffer);
    
    return result;
}

/* Get system memory info */
static PyObject* get_memory_info(PyObject* self, PyObject* args) {
    // Simplified memory info - in production, use system-specific APIs
    PyObject* dict = PyDict_New();
    
    PyDict_SetItemString(dict, "total_mb", PyLong_FromLong(512));
    PyDict_SetItemString(dict, "available_mb", PyLong_FromLong(256));
    PyDict_SetItemString(dict, "usage_percent", PyFloat_FromDouble(50.0));
    
    return dict;
}

/* Check text encoding */
static PyObject* check_encoding(PyObject* self, PyObject* args) {
    const char* text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }
    
    // Check for valid UTF-8
    int valid = 1;
    for (int i = 0; text[i] != '\0'; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c > 127) {  // Non-ASCII character
            // Basic UTF-8 check
            if ((c & 0xE0) == 0xC0) {
                if ((unsigned char)text[i+1] >> 6 != 2) valid = 0;
                i++;
            } else if ((c & 0xF0) == 0xE0) {
                if ((unsigned char)text[i+1] >> 6 != 2) valid = 0;
                if ((unsigned char)text[i+2] >> 6 != 2) valid = 0;
                i += 2;
            }
        }
    }
    
    return PyBool_FromLong((long)valid);
}

/* Method definitions */
static PyMethodDef SystemOpsMethods[] = {
    {"get_timestamp", get_timestamp, METH_NOARGS, "Get current timestamp"},
    {"fast_hash", fast_hash, METH_VARARGS, "Fast string hashing"},
    {"text_complexity", text_complexity, METH_VARARGS, "Calculate text complexity"},
    {"optimize_text", optimize_text, METH_VARARGS, "Optimize text by removing extra spaces"},
    {"get_memory_info", get_memory_info, METH_NOARGS, "Get system memory information"},
    {"check_encoding", check_encoding, METH_VARARGS, "Check if text is valid UTF-8"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef system_ops_module = {
    PyModuleDef_HEAD_INIT,
    "system_ops",
    "System operations extension module for AI Chatbot",
    -1,
    SystemOpsMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_system_ops(void) {
    return PyModule_Create(&system_ops_module);
}
