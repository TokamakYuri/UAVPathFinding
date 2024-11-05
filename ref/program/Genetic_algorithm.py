import numpy as np
import copy
import time
from copy import deepcopy
import math
import matplotlib.pyplot as plt

from profit_matrix import profit_matrix


def create_population(size, num_tasks):
    return [np.random.permutation(num_tasks) for _ in range(size)]


def fitness(individual, rewards):
    total_reward = 0
    for i, task in enumerate(individual):
        total_reward += rewards[i][task]
    return total_reward


def select(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population) // 2):
        tournament = np.random.choice(len(population), tournament_size, replace=False)
        best = max(tournament, key=lambda i: fitnesses[i])
        selected.append(population[best])
    return selected


def crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size
    crossover_point1, crossover_point2 = sorted(np.random.choice(size, 2, replace=False))
    child1[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
    child2[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]

    fill_pos1, fill_pos2 = crossover_point2, crossover_point2
    for i in range(size):
        if parent2[i] not in child1:
            while child1[fill_pos1 % size] != -1:
                fill_pos1 += 1
            child1[fill_pos1 % size] = parent2[i]
        if parent1[i] not in child2:
            while child2[fill_pos2 % size] != -1:
                fill_pos2 += 1
            child2[fill_pos2 % size] = parent1[i]
    return child1, child2


def mutate(individual):
    idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def genetic_algorithm(rewards, population_size, max_generations, target_fitness, tolerance, crossover_prob=0.8,
                      mutation_prob=0.05):
    num_tasks = len(rewards)
    population = create_population(population_size, num_tasks)
    best_fitness = -float('inf')

    fitness_over_time = []  # 记录每一代的误差

    for generation in range(max_generations):
        fitnesses = [fitness(ind, rewards) for ind in population]
        best_fitness_current = max(fitnesses)

        # 记录误差
        error_gen = abs(best_fitness_current - target_fitness)
        if error_gen < 3:
            fitness_over_time.append(error_gen)

        if abs(best_fitness_current - target_fitness) <= tolerance:
            break

        if best_fitness_current > best_fitness:
            best_fitness = best_fitness_current

        selected = select(population, fitnesses)
        next_population = []
        while len(next_population) < population_size:
            parents = np.random.choice(len(selected), 2, replace=False)
            if np.random.rand() < crossover_prob:
                child1, child2 = crossover(selected[parents[0]], selected[parents[1]])
            else:
                child1, child2 = selected[parents[0]], selected[parents[1]]
            if np.random.rand() < mutation_prob:
                child1 = mutate(child1)
            if np.random.rand() < mutation_prob:
                child2 = mutate(child2)
            next_population.extend([child1, child2])
        population = next_population[:population_size]

    best_individual = max(population, key=lambda ind: fitness(ind, rewards))
    return best_individual, fitness(best_individual, rewards), fitness_over_time


def has_duplicates(matrix):
    seen = set()
    for row in matrix:
        if row in seen:
            return True
        seen.add(row)
    return False


# 示例参数
A = copy.deepcopy(profit_matrix)
profits = np.array(A)

start = time.time()
target_fitness = 609.705627
best_assignment, best_reward, fitness_over_time = genetic_algorithm(
    profits,  # 收益矩阵
    population_size=1000,  # 种群规模
    max_generations=10000,  # 最大代数
    target_fitness=609.705627,  # 目标收益值
    tolerance=0.1,  # 允许的误差范围
    crossover_prob=0.4,  # 交叉概率
    mutation_prob=0.005  # 变异概率
)

end = time.time()

print("分配方案中是否有重复元素:", has_duplicates(best_assignment))
print("最佳任务分配方案:\n", best_assignment)
print("最佳总收益:", best_reward)
print("算法运行时间：", (end - start))
print("误差为：", (target_fitness - best_reward))

# 绘制误差变化曲线
plt.plot(range(len(fitness_over_time)), fitness_over_time)
plt.xlabel('Iterations')
plt.ylabel('Error')
#plt.title('Error vs Generations')
plt.grid(True)
plt.show()
