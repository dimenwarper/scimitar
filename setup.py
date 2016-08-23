from setuptools import setup

setup(name='scimitar',
        version='0.1',
        description='Trajectory analysis of single cell measurements',
        author='Pablo Cordero',
        author_email='dimenwarper@gmail.com',
        url='https://github.com/dimenwarper/scimitar',
        packages=['scimitar'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'networkx', 'matplotlib', 'pandas', 'seaborn', 'munkres']
        )
