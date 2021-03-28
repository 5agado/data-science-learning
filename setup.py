from setuptools import setup

setup(
    name="data_science_learning",
    version="0.1.1",
    author="Alex Martinelli",
    description="Utils library for Data-Science related projects",
    license="Apache 2.0",
    packages=['ds_utils', 'scripts'],
    entry_points={
        'console_scripts': ['convert-video=scripts.convert_video:main',
                            'video-mosaic=scripts.video_mosaic:main',
                            'image-montage=scripts.image_montage:main'],
    },
    install_requires=[
        'numpy',
        'requests',
        'pandas',
        'matplotlib',
        'seaborn',
        'imageio',
        'opencv-python',
        'scipy',
        'tqdm',
        'pyyaml',
        'plotly',
        'cufflinks',
        'imageio-ffmpeg',
        'pillow',
        'ipywidgets',
        'jupyter_contrib_nbextensions',
    ],
)