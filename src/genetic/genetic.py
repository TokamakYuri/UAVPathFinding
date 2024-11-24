import numpy as np
import numpy.random as nprng
from numpy import (
    inf
)

import utility.utility as ut

rng = nprng.default_rng()

class individual:
    def __init__(self, pathnum) -> None:
        self.pos = np.zeros((3, pathnum))
        self.fitness = inf
        self.path = None

def create(pathnum, pop):
    population = np.array([])
    for i in range(pop):
        population = np.append(population, [individual(pathnum)], axis=0)
    return population

def updateFitness(population : np, weight : dict, settings : dict, radar : np, radarsettings : dict, terrain : np = None):
    for i in range(population.shape[0]):
        population[i].fitness = ut.calaFitness(population[i].path, weight, settings, radar, radarsettings, terrain)
    return population

def initPopulation(population : np, pathnum : int, posbound : np, start : np, stop : np, rng : nprng = rng):
    size = population.shape[0]
    for i in range(size):
        for j in range(3):
            population[i].pos[j] = rng.uniform(posbound[0][j], posbound[1][j], pathnum)
        population[i].pos = np.sort(population[i].pos)
        population[i].path = np.array([start])
        arr = np.rot90(population[i].pos, -1)
        for ele in arr:
            population[i].path = np.append(population[i].path, [ele], axis=0)
        population[i].path = np.append(population[i].path, [stop], axis=0)
    return population

def updatePopulationPath(population : np, start : np, stop : np):
    size = population.shape[0]
    for i in range(size):
        population[i].path = np.array([start])
        arr = np.rot90(population[i].pos, -1)
        for ele in arr:
            population[i].path = np.append(population[i].path, [ele], axis=0)
        population[i].path = np.append(population[i].path, [stop], axis=0)
    return population

def sortPopulation(population : np):
    size = population.shape[0]
    for i in range(size):
        for j in range(size - i - 1):
            if population[i].fitness > population[i + j].fitness:
                tmp = population[i]
                population[i] = population[i + j]
                population[i + j] = tmp
    return population

def selection(population : np, rng=rng):
    # size = population.shape[0]
    # lm = np.arange(size) / size
    # r = rng.random(size)
    # select = np.less(lm, r)
    # repop = np.array([])
    # for i in range(size):
    #     if select[i]:
    #         repop = np.append(repop, [population[i]], axis=0)
    # return repop
    repop = np.split(population, 2)
    return repop[0]

    
def crossover(population : np):
    size = population.shape[0]
    newpop = np.array([])
    for i in range(int(size / 2.)):
        par_a = population[2 * i]
        par_b = population[2 * i + 1]
        newpop = np.append(newpop, [par_a, par_b], axis=0)
        pathnum = par_a.pos[0].shape[0]
        rand = nprng.randint(pathnum)
        arr_a = np.hsplit(par_a.pos, [rand])
        arr_b = np.hsplit(par_b.pos, [rand])
        par_a.pos = np.hstack((arr_a[0], arr_b[1]))
        par_b.pos = np.hstack((arr_b[0], arr_a[1]))
        newpop = np.append(newpop, [par_a, par_b], axis=0)
    return newpop

def mutation(population : np, posbound : np, rng=rng):
    size = population.shape[0]
    newpop = np.array([])
    for i in range(size):
        indi = population[i]
        rand = nprng.randint(indi.pos[0].shape[0])
        indi.pos = np.delete(indi.pos, rand, 1)
        pos = np.zeros((3, 1))
        for j in range(3):
            pos[j] = rng.uniform(posbound[0][j], posbound[1][j])
        indi.pos = np.sort(np.hstack((indi.pos, pos)))
        newpop = np.append(newpop, [indi], axis=0)
    return newpop
        
def produce(population : np, posbound : np, pathnum : int, rng=rng):
    try:
        parent = np.split(population, 2)[0]
    except:
        print(1)
    descendant = np.array([])
    size = parent.shape[0]
    for i in range(int(size / 2)):
        parent_a = parent[i]
        parent_b = parent[size - i - 1]
        rand = nprng.randint(pathnum)
        pos_a = np.hsplit(parent_a.pos, [rand])
        pos_b = np.hsplit(parent_b.pos, [rand])
        des_ap = np.hstack((pos_a[0], pos_b[1]))
        des_bp = np.hstack((pos_b[0], pos_a[1]))
        des_ap = np.delete(des_ap, nprng.randint(pathnum), 1)
        des_bp = np.delete(des_bp, nprng.randint(pathnum), 1)
        mut_ap = np.zeros((3, 1))
        mut_bp = np.zeros((3, 1))
        for j in range(3):
            mut_ap[j] = rng.uniform(posbound[0][j], posbound[1][j])
            mut_bp[j] = rng.uniform(posbound[0][j], posbound[1][j])
        des_ap = np.sort(np.hstack((des_ap, mut_ap)))
        des_bp = np.sort(np.hstack((des_bp, mut_bp)))
        des_a = individual(pathnum)
        des_b = individual(pathnum)
        des_a.pos = des_ap
        des_b.pos = des_bp
        descendant = np.append(descendant, [des_a, des_b], axis=0)
    newpop = np.append(parent, descendant)
    return newpop

def genetic(start : np, stop : np, posbound : np, geneoptions : dict, terrainsettings : dict, radar : np, radarsettings : dict, terrain : np = None, rng : nprng = rng):
    indinum = geneoptions.get('indinum')
    gen = geneoptions.get('gen')
    pathnum = geneoptions.get('pathnum')
    weight = geneoptions.get('weight')
    populations = create(pathnum, indinum)
    for i in range(gen):
        if i == 0:
            initPopulation(populations, pathnum, posbound, start, stop, rng)
        populations = updatePopulationPath(populations, start, stop)
        populations = updateFitness(populations, weight, terrainsettings, radar, radarsettings)
        populations = sortPopulation(populations)
        populations = produce(populations, posbound, pathnum, rng)
        # populations = selection(populations, rng)
        # populations = crossover(populations)
        # populations = mutation(populations, posbound, rng)
        if i % 100 == 0:
            print(str(i) + ':' + str(populations[0].fitness))
    return populations