from distutils.core import setup

setup(
    name='snowflake',
    version='0.1.0',
    url='github.com/ucb-sejits/snowflake',
    license='B',
    author='Nathan Zhang',
    author_email='nzhang32@berkeley.edu',
    description='DSL/IR for Stencils in Python',

    packages=[
        'snowflake'
    ],

    install_requires=[
        'sympy',
        'numpy',
        'ctree',
        'rebox',
        'sphinx'
    ]
)
