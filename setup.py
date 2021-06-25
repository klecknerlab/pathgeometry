from setuptools import setup

setup(
    name='pathgeometry',
    version='0.1',
    description='Library computing geometrical quantites on 3D path',
    url='https://github.com/klecknerlab/pathgeometry',
    author='Dustin Kleckner',
    author_email='dkleckner@ucmerced.edu',
    license='Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)',
    packages=['pathgeometry'],
    install_requires=[
        'numpy',
        'scipy'
    ],
    # scripts=['bin/muvi_convert'],
    # entry_points={
    #     'gui_scripts': ['muvi=muvi.view.qtview:qt_viewer']
    # },
)
