from setuptools import setup, Extension
from torch.utils import cpp_extension

# python setup.py --dry-run build
# python setup.py --help

setup(
    # refer to https://docs.python.org/3/distutils/setupscript.html#additional-meta-data
    name='my-hello',
    version='1.0.2',
    author='foo',
    author_email='foo@bar.com',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='hello',
            sources=['hello.cc'],
            extra_compile_args=['-Wno-unknown-pragmas'],  # ['-g']
            include_dirs=['/tmp/foo'],  # ['abc', '/tmp/foo']
            define_macros=[
                ('FOO', None),  # -DFOO
                ('BAR', 23),  # -DBAR=23
            ],
            undef_macros=[
                'HAVE_FOO',  # -UHAVE_FOO
            ],
            library_dirs=['/tmp/bar'],  #[]
            runtime_library_dirs=['/tmp/rpath'],
            libraries=[],  # it will add in the code: c10, torch, torch-cpu, torch-python
            # for cuda, it adds additionaly: cudart, c10_cuda, torch_cuda
            language='c++',  # it is hardcoded in the code.
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)},
    scripts=['./abc.sh'],
    data_files=[
        ('my_data', ['./Makefile', './hello.cc']),  # it will create a directory `my_data`
        ('', ['./setup.py', './Makefile']),  # it puts `./abc.txt` and `./a.txt` in the package dir
    ],
)
