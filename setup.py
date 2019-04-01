from setuptools import setup

setup(
    name="data_science_learning",
    version="0.1.1",
    author="Alex Martinelli",
    description="Utils library for Data-Science related projects",
    license="Apache 2.0",
    packages=['ds_utils', 'scripts'],
    entry_points={
        'console_scripts': ['convert-video=scripts.convert_video:main'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn',
        'opencv-python',
        'scipy',
        'keras',
        'tqdm',
    ],
)