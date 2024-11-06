import numpy as np
import numpy.random as nprng

from numpy import inf

import utility.utility as ut

rng = nprng.default_rng()

class particle:
    def __init__(self) -> None:
        self.pos = np.array([[]])
        self.fitness = inf
        self.path = np.array([])
        self.vel = np.array([[]])
        self.bestpos = np.array([[]])
        self.bestfitness = inf
        self.bestpath = np.array([])
    
def create(num : int) -> np:
    particles = np.array([])
    for i in range(num):
        particles = np.append(particles, [particle()], axis = 0)
    return particles

def updateFitness(particles : np, weight : dict, settings : dict, terrain : np = None):
    for i in range(particles.shape[0]):
        particles[i].fitness = ut.calaFitness(particles[i].path, weight, settings, terrain)
    return particles
    
def initParticles(particles : np, pathnum : int, posbound : np, velbound : np, start : np, stop : np):
    size = particles.shape[0]
    for i in range(size):
        for j in range(3):
            particles[i].pos = np.append(
                particles[i].pos, 
                rng.uniform(posbound[0][j], posbound[1][j], pathnum), 
                axis = 0
            )
            particles[i].vel = np.append(
                particles[i].vel,
                rng.uniform(velbound[0][j], velbound[1][j], pathnum),
                axis = 0
            )
            if j != 2:
                particles[i].pos[j] = np.sort(particles[i].pos[j])
        particles[i].path = np.append(particles[i].path, [start], axis = 0)
        particles[i].path = np.append(particles[i].path, ut.rotate90MatUP(particles[i].pos, -1), axis = 0)
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
            particles[i].bestpos = particles[i].pos
            particles[i].bestpath = particles[i].path
    return particles

def updateParticlesPath(particles : np, start : np, stop : np):
    for i in range(particles.shape[0]):
        array = np.array([])
        array = np.append(array, [start], axis = 0)
        array = np.append(array, ut.rotate90MatUP(particles[i].pos, -1), axis = 0)
        array = np.append(array, [stop], axis = 0)
        particles[i].path = array
    return particles

def updateParticlesVelocity(partucles : np, globalbest : particle, options : dict, velbound : np, pathnum : int):
    w = options.get('w')
    c1 = options.get('c1')
    c2 = options.get('c2')
    for i in range(partucles.shape[0]):
        for j in range(3):
            partucles[i].vel[j] = w * partucles[i].vel[j]
            + c1 * rng.uniform(velbound[0][j], velbound[1][j], pathnum) * (partucles[i].bestpos[j] - partucles[i].pos[j])
            + c2 * rng.uniform(velbound[0][j], velbound[1][j], pathnum) * (globalbest.bestpos[j] - partucles[i].pos[j])
            partucles[i].pos[j] = np.add(partucles[i].pos[j], partucles[i].vel[j])
    return partucles
    
def pso(start : np, stop : np, posbound : np, velbound : np, weight : dict, psooption : dict, terrainsettings : dict, terrain : np = None):
    particlenum = psooption.get('num')
    step = psooption.get('step')
    pathnum = psooption.get('pathnum')
    particles = create(particlenum)
    globalbest = particle()
    for i in range(step):
        if i == 0:
            particles = initParticles(particles, pathnum, posbound, velbound, start, stop)
        else:
            particles = updateParticlesVelocity(particles, globalbest, psooption, velbound, pathnum)
        particles = updateParticlesPath(particles, start, stop)
        particles = updateFitness(particles, weight, terrainsettings, terrain)
        particles = updateLocalBestParticles(particles)
        globalbest = updateGlobalBestParticles(particles, globalbest)
        if i % 100 == 0:
            print(globalbest.bestfitness)
    return globalbest