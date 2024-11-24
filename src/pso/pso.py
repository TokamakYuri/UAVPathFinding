import numpy as np
import numpy.random as nprng

from numpy import inf

import utility.utility as ut

rng = nprng.default_rng()

class particle:
    def __init__(self) -> None:
        self.pos = np.array([])
        self.fitness = inf
        self.path = np.array([])
        self.vel = np.array([])
        self.bestpos = np.array([])
        self.bestfitness = inf
        self.bestpath = np.array([])
    
def create(num : int) -> np:
    particles = np.array([])
    for i in range(num):
        particles = np.append(particles, [particle()], axis = 0)
    return particles

def updateFitness(particles : np, weight : dict, settings : dict, radar : np, radarsettings : dict, terrain : np = None):
    for i in range(particles.shape[0]):
        particles[i].fitness = ut.calaFitness(particles[i].path, weight, settings, radar, radarsettings, terrain)
    return particles
    
def initParticles(particles : np, pathnum : int, posbound : np, velbound : np, start : np, stop : np, rng : nprng = rng):
    size = particles.shape[0]
    for i in range(size):
        particles[i].pos = np.empty((3, pathnum))
        particles[i].vel = np.empty((3, pathnum))
        for j in range(3):
            particles[i].pos[j] = rng.uniform(posbound[0][j], posbound[1][j], pathnum)
            particles[i].vel[j] = rng.uniform(velbound[0][j], velbound[1][j], pathnum)
            particles[i].pos[j] = np.sort(particles[i].pos[j])
            # if i != 2:
                # particles[i].pos[j] = np.sort(particles[i].pos[j])
        particles[i].path = np.array([start])
        arr = np.rot90(particles[i].pos, -1)
        for ele in arr:
            particles[i].path = np.append(particles[i].path, [ele], axis = 0)
        particles[i].path = np.append(particles[i].path, [stop], axis = 0)
    return particles

def updateGlobalBestParticles(particles : np, globalbest : particle):
    for i in range(particles.shape[0]):
        if particles[i].bestfitness < globalbest.bestfitness:
            globalbest = particles[i]
    return globalbest

def updateLocalBestParticles(particles : np):
    for i in range(particles.shape[0]):
        if particles[i].bestfitness > particles[i].fitness:
            particles[i].bestfitness = particles[i].fitness
            particles[i].bestpos = np.copy(particles[i].pos)
            particles[i].bestpath = np.copy(particles[i].path)
    return particles

def updateParticlesPath(particles : np, start : np, stop : np):
    for i in range(particles.shape[0]):
        array = np.array([start])
        tmp = np.rot90(particles[i].pos, -1)
        for ele in tmp:
            array = np.append(array, [ele], axis = 0)
        array = np.append(array, [stop], axis = 0)
        particles[i].path = array
    return particles

def updateParticlesVelocity(particles : np, globalbest : particle, options : dict, velbound : np, posbound : np, pathnum : int, rng : nprng, w : int = None):
    if w == None:
        w = options.get('w')
    c1 = options.get('c1')
    c2 = options.get('c2')
    for i in range(particles.shape[0]):
        for j in range(3):
            particles[i].pos[j] = np.clip(np.add(particles[i].pos[j], particles[i].vel[j]), posbound[0][j], posbound[1][j])
            particles[i].vel[j] = w * particles[i].vel[j] + c1 * rng.uniform(0, 1, pathnum) * (particles[i].bestpos[j] - particles[i].pos[j]) + c2 * rng.uniform(0, 1, pathnum) * (globalbest.bestpos[j] - particles[i].pos[j])
            particles[i].vel[j] = np.clip(particles[i].vel[j], velbound[0][j], velbound[1][j])
    return particles
    
def pso(start : np, stop : np, posbound : np, velbound : np, psooption : dict, terrainsettings : dict, radar : np, radarsettings : dict, terrain : np = None, rng : nprng = rng):
    particlenum = psooption.get('num')
    step = psooption.get('step')
    pathnum = psooption.get('pathnum')
    weight = psooption.get('weight')
    particles = create(particlenum)
    globalbest = particle()
    for i in range(step):
        if i == 0:
            particles = initParticles(particles, pathnum, posbound, velbound, start, stop, rng)
        else:
            particles = updateParticlesVelocity(particles, globalbest, psooption, velbound, posbound, pathnum, rng)
        particles = updateParticlesPath(particles, start, stop)
        particles = updateFitness(particles, weight, terrainsettings, radar, radarsettings, terrain)
        particles = updateLocalBestParticles(particles)
        globalbest = updateGlobalBestParticles(particles, globalbest)
        if i % 100 == 0:
            print(str(i) + ':' + str(globalbest.bestfitness))
    return globalbest

def decWeight(weibound : np, gen : int, maxgen : int):
    p = -(gen / maxgen) ** 2 + 1
    w = weibound[0] - (weibound[1] - weibound[0]) * p
    return w

def ldpso(start : np, stop : np, posbound : np, velbound : np, weibound : np, psooption : dict, terrainsettings : dict, radar : np, radarsettings : dict, terrain : np = None, rng : nprng = rng):
    particlenum = psooption.get('num')
    step = psooption.get('step')
    pathnum = psooption.get('pathnum')
    weight = psooption.get('weight')
    particles = create(particlenum)
    globalbest = particle()
    for i in range(step):
        w = decWeight(weibound, i, step)
        if i == 0:
            particles = initParticles(particles, pathnum, posbound, velbound, start, stop, rng)
        else:
            particles = updateParticlesVelocity(particles, globalbest, psooption, velbound, posbound, pathnum, rng, w=w)
        particles = updateParticlesPath(particles, start, stop)
        particles = updateFitness(particles, weight, terrainsettings, radar, radarsettings, terrain)
        particles = updateLocalBestParticles(particles)
        globalbest = updateGlobalBestParticles(particles, globalbest)
        if i % 100 == 0:
            string = str(i) + ':' + str(globalbest.bestfitness)
            print(string)
    return globalbest