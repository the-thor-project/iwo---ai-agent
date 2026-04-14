"""
Setup script for building C extension module
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import platform

# C Extension module
system_ops_ext = Extension(
    'system_ops',
    sources=['system_ops.c'],
    extra_compile_args=[
        '/O2' if platform.system() == 'Windows' else '-O2',
        '/W4' if platform.system() == 'Windows' else '-Wall'
    ]
)

setup(
    name='AI Chatbot C Extension',
    version='1.0.0',
    description='C extension for system operations in AI Chatbot',
    ext_modules=[system_ops_ext],
    python_requires='>=3.8'
)
