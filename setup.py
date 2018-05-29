from setuptools import setup

setup(
    name="data_science_utils",
    version="0.1.0",
    author="Alex Martinelli",
    description="Utils library for Data-Science related projects",
    license="Apache 2.0",
    packages=['utils'],
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