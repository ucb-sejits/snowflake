from distutils.core import setup

setup(
    name='stencil',
    version='0.1.0',
    url='github.com/ucb-sejits/stencil',
    license='B',
    author='Nathan Zhang',
    author_email='nzhang32@berkeley.edu',
    description='DSL/IR for Stencils in Python',

    packages=[
        'stencil'
    ],

    install_requires=[
        'numpy',
        'ctree',
        'rebox'
    ]
)
