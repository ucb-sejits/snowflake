from snowflake._utils import make_enum

__author__ = 'nzhang-dev'

OptimizationLevels = make_enum(
    'OptimizationLevels',
    [
        'SNOWFLAKE',
        'REIFIED',
        'PRECOMPILED'
    ]
)