from setuptools import setup, find_packages


setup(
    name="face_extract",
    version="0.1.0",
    author="Alex Martinelli",
    description="Face extract tools and utils",
    license="Apache 2.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['extract=face_utils.extract:main'],
    },
    install_requires=[
        'cmake',  # should be installed before dlib
        'numpy',
        'pandas',
        'matplotlib',
        'opencv-python',
        'scipy',
        'tensorflow-gpu',
        'keras',
        'scikit-image',
        'tqdm',
        'dlib',
        'mtcnn', #  https://github.com/ipazc/mtcnn
    ],
)