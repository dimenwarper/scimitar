from setuptools import setup

setup(name='scimitar',
        version='0.1',
        description='Trajectory analysis of single cell measurements',
        author='Pablo Cordero',
        author_email='dimenwarper@gmail.com',
        url='https://github.com/dimenwarper/scimitar',
        packages=['coexpression', 'model_comparisons', 'plotting', 'simulation', 'differential_analysis', 'models', 'preprocessing', 'stats', 'morphing_mixture', 'principal_curves', 'utils', 'log', 'pipelines', 'settings'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'networkx', 'matplotlib', 'pandas', 'seaborn', 'munkres']
        )
