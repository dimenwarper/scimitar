import numpy as np

def samples_to_tsv(fl):
    matrix = np.load(fl)
    outfile = open(fl.name.replace('.npy', '.tsv'), 'w')
    outfile.write('Sample\t' + '\t'.join(['Gene%s' % i for i in xrange(matrix.shape[1])]) + '\n')
    for i in xrange(matrix.shape[0]):
        outfile.write('Sample%s\t' % i + '\t'.join([str(x) for x in matrix[i, :]]) + '\n')
    outfile.close()

samples_to_tsv(open('path_1_samples.npy'))
samples_to_tsv(open('path_2_samples.npy'))

