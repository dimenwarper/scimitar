<img src="https://lh3.googleusercontent.com/dqsSqpn5TyjoFZimQC5PIDPJt9kN965o_Nbi25N_JrOz31TRzkl3ce272sAGLdJdY9zvSppxbdtq0_c=w1510-h786-rw" width="300">
## The Single Cell Inference of MorphIng Transitions and Associated Regulation module

SCIMITAR provides a variety of tools to analyze trajectory maps of single-cell measurements. 

With SCIMITAR you can:
* Obtain coarse-grain, (metastable) state and transition representations of your data. This is useful when you want to get a broad sense of how your data is connected.
* Infer full-fledged Gaussian distribution trajectories from single-cell data --- not only will you get cell orderings and estiamted 'pseudotemporal' mean measurements but also pseudo-time-dependant covariance matrices so you can track how your measurements' correlation change across biological progression.
* Obtain uncertainties for a cell's psuedotemporal positioning (due to uncertainty arising from heteroscedastic noise)
* Obtain genes that significantly change throughout the progression (i.e. 'progression-associated genes')
* Obtain genes that significantly change their correlation structure throughout the progression (i.e. 'progression co-associated genes')
* Infer broad co-regulatory states and psuedotemporal dynamic gene modules from the evolving co-expression matrices.


To install SCIMITAR, follow the steps below:

1. Install the [pyroconductor](https://github.com/dimenwarper/pyroconductor) package 

2. Do the usual `python setup.py install`

3. Check out the jupyter notebooks tutorials in the tutorials directory

4. Questions, concerns, or suggestions? Thanks! Open up a ticket or pm [@dimenwarper](https://github.com/dimenwarper) (Pablo Cordero)
