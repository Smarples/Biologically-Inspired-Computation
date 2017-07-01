#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates a neural network and trains it using a hill-climbing evolutionary algorithm.
    Network then trained to predict fitness of function, then used as surrogate function to 
    attempt to optimise function to target fitness.

Under unix-like systems: 
    nohup nice python neuroevolutionexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks

argv = sys.argv[1:] # shortcut for input arguments

datapath = 'NEUROEVOLUTION' if len(argv) < 1 else argv[0]

dimensions = [2,3,5,10,20,40] if len(argv) < 2 else eval(argv[1])
function_ids = [1] if len(argv) < 3 else eval(argv[2])  
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 3) + range(41, 43) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='Neuroevolution',
            comments='3 layer artifical network learns through hill-climbing EA. Used as surrogate to optimise continuous functions.')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes 
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic 


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """ Runs the neuroevolution optimiser
    """
    NEUROEVOLUTION(fun, dim,  maxfunevals, ftarget)


	
def NEUROEVOLUTION(fun,dim, maxfunevals, ftarget):
    """creates training inputs in dim^[-5,5] and calculates desired output using fun()
    trains 3-layer artificial neural network, evolving with hill climbing EA. 
    once trained, uses genetic algorithm to find optimal inputs for target fitness
    and compares ANN fitness to CoCo fun() fitness. returns ANN fitness for post-processing

    """
    # implementation of the sphere function - NOT USED
    #def sphere(x):
        #val = 0
        #for i in range(0, len(x)):
            #val += (int(x[i])^2)
        #return val
    
    # tanh : activation function
    # tanh(x) = (2 / 1 + e^(-2x)) - 1
    def activation(x):
	return ( 2 / ( 1 + np.exp(-2 * x))) - 1
    
    # evaluates chromsome by feeding through ANN
    def annEval(chrm, anninput):
        # initialises first synapse with associated weights in chromosome
        syn0 = np.zeros((dim, hiddenc))
        for i in range(0, dim):
            for j in range(0, hiddenc):
                syn0[i][j] = chrm[i*hiddenc+j]
        # initialises second synapse with remaining weights
        syn1 = chrm[chrmlen - hiddenc:chrmlen]
        # sets the input layer of ANN to activated inputs
        layerin = anninput
        # hidden layer
        layerhid = activation(np.dot(layerin, syn0))
        # output layer
        layerout = activation(np.dot(layerhid, syn1))
        return layerout
    
    # mutates provided chromosome with probability 'rate'
    def tmutate(chrm, rate):
        # for each weight in the chromosome
        for w in range(0, len(chrm)):
            # if the weight is to be mutated
            if np.random.rand() < rate:
                # mutates weight by +/-0.5 - NOT USED
                #chrm[w] = (chrm[w] + 1 * np.random.rand() - 0.5) % 1
                # mutates weight to new random weight
                chrm[w] = 2 * np.random.rand() - 1
        return chrm
        
    # number of neurons in hidden layer
    hiddenc = 20
    # length of the chromosome (no. of weights in the neural network)
    chrmlen = dim * hiddenc + hiddenc
    
    # training population size
    tpopsize = 200
    # training generations
    tgens = 20
    # training instances count
    tcount = 100
    # training mutation rate
    tmutation = 0.1
    # training inputs
    tinput = 10 * np.random.rand(tcount, dim) - 5
    # training outputs from CoCo
    toutput = fun(tinput)
    # training outputs from sphere() - NOT USED
    #toutput = np.zeros(tcount)
    #for i in range(0, tcount):
        #toutput[i] = sphere(tinput[i])

    print "-" * 100
    # initialise training population
    print "Initialising training data..."
    tpop = 2 * np.random.rand(tpopsize, chrmlen) - 1
    print tcount, "sets of training data initialised."
    print ""
    
    print "Beginning training..."
    # initialise array of errors to track average
    terror = np.zeros(tcount)
    # create blank chromosome (neural network weights)
    nn = np.zeros(chrmlen)
    # for each training pattern
    for pat in range(0, tcount):
        #print "Training Pattern", pat
        #print "Input", tinput[pat]
        #print "Expected Fitness", toutput[pat]
        # tracks fitness of fittest in the population
        ebest = np.inf
        # holds weights of fittest in population
        pbest = np.zeros(chrmlen)
        
        # for each generation
        for gen in range(0, tgens):
            # for each member of population
            for mem in range(0, tpopsize):
                # evaluate ANN error (* 1000 to scale)
                annerror = abs(toutput[pat] - (annEval(tpop[mem], tinput[pat]) * 1000))
                # if error is less than best (higher fitness)
                if annerror < ebest:
                    # update best fitness and best chromosome
                    ebest = annerror
                    pbest = tpop[mem]
                    # update neural network weights
                    nn = pbest.copy()
            #print "Generation", gen
            #print "Smallest Error", ebest
            
            # hill climbing - keep highest fitness individual
            tpop[0] = pbest.copy()
            # for all other individuals
            for mem in range(1, tpopsize):
                # replace populations with mutations of rate tmutation
                tpop[mem] = tmutate(tpop[0].copy(), tmutation)
                
        # add error to error array
        terror[pat] = ebest
        # print to console after every 10 patterns
        if pat % 10 == 0 and pat != 0:
            print "  ", pat, "patterns learned | Average error:", np.average(terror[0:pat])
            
    # training complete
    print "Training complete.", tcount, "patterns learned with average error", np.average(terror)
    print ""

    # use neural network as surrogate function to optimise fitness
    # find optimal inputs with genetic algorithm w/ elitism and mutation
    print "Optimising inputs with ANN..."
    print ""
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    elitism = 10
    mutation = 0.1
    fbest = np.inf
    # generate random initial population
    xpop = 10 * np.random.rand(int(popsize), dim) - 5
    
    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        # evaluate fitness of population through ANN
        fvalues = annEval(nn, xpop) * 1000
        idx = np.argsort(fvalues)
        # new best fitness found
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved
            print "Task achieved:"
            break
        # initialise empty temporary population
        tmppop = np.zeros(popsize, dim)
        # for elite (fittest) individuals
        for i in range(0, elitism):
            # add to new temporary population
            tmppop[i] = xpop[idx[i]]
        # replace rest of population with mutations of rate 'mutation'
        for j in range(elitism, popsize):
            if np.random.rand() < mutation:
                tmppop[j] = 10 * np.random.rand(dim) - 5
            else:
                tmppop[j] = xpop[j]
        # update population
        xpop = tmppop.copy()
    print "  Target fitness:", ftarget
    print "  Optimised inputs:", xbest
    print "  ANN predicted output:", (annEval(nn, xbest) * 1000)
    print "  CoCo actual output:", fun(xbest)
    print "  Error of ANN:", abs((annEval(nn, xbest) * 1000) - fun(xbest))
    print ""
    # return best fitness inputs
    return xbest

    
# neuroevolution algorithm finished


# set random seed
t0 = time.time()
np.random.seed(int(t0))
#np.random.seed(0)


f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            f.finalizerun()
            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim
