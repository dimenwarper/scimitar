<img src="https://github.com/dimenwarper/scimitar/raw/master/logo.png" width="300">
## Single Cell Inference of MorphIng Trajectories and their Associated Regulation module

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


If you use SCIMITAR please cite the [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5203771/) ;)

Cordero and Stuart, "Tracing co-regulatory network dynamics in noisy, single-cell transcriptome trajectories", Pac. Symp. of Biocomput. (2017)
