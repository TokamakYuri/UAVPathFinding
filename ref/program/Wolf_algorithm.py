import numpy as np
import copy
import time

# 你的收益矩阵
from profit_matrix import profit_matrix

# 计算任务分配的总收益
def calculate_fitness(assignments, profits):
    profits = np.array(profits)  # 确保 profits 是 NumPy 数组
    return np.sum([profits[i, assignments[i]] for i in range(len(assignments))])


# 初始化狼群,返回num_wolves个解
def initialize_wolves(num_wolves, num_tasks):
    return [np.random.permutation(num_tasks) for _ in range(num_wolves)]


# 确保分配方案是有效的排列
def repair_solution(solution):
    _, indices = np.unique(solution, return_inverse=True)
    return np.argsort(indices)


# 更新狼的位置
def update_wolves(wolves, alpha, beta, delta, profits, scout_step_length, alpha_step_length, scout_ratio, update_ratio):
    new_wolves = []
    num_wolves = len(wolves)
    num_tasks = len(alpha)

    for wolf in wolves:
        new_wolf = wolf.copy()
        for i in range(num_tasks):
            rand_val = np.random.rand()
            if rand_val < scout_ratio:
                new_wolf[i] = int(wolf[i] + np.random.uniform(-scout_step_length, scout_step_length))       # 探狼随机步长
            elif rand_val < scout_ratio + update_ratio:
                new_wolf[i] = int(alpha[i] + np.random.uniform(-alpha_step_length, alpha_step_length))      # 头狼根据alpha位置更新
            else:
                new_wolf[i] = int(beta[i] + np.random.uniform(-alpha_step_length, alpha_step_length))       # 猛狼根据beta位置更新

        # 修复位置
        new_wolf = repair_solution(new_wolf)
        new_wolves.append(new_wolf)
    return new_wolves


def wolf_pack_algorithm(num_wolves, num_tasks, profits, max_iter, target_fitness, tolerance, scout_step_length,
                        alpha_step_length, scout_ratio, update_ratio):
    # 初始化
    wolves = initialize_wolves(num_wolves, num_tasks)
    alpha = beta = delta = wolves[0]
    alpha_fitness = beta_fitness = delta_fitness = -float('inf')

    for _ in range(max_iter):
        for wolf in wolves:
            fitness = calculate_fitness(wolf, profits)      # 计算适应度
            if fitness > alpha_fitness:
                delta, beta, alpha = beta, alpha, wolf
                alpha_fitness = fitness
            elif fitness > beta_fitness:
                delta, beta = beta, wolf
                beta_fitness = fitness
            elif fitness > delta_fitness:
                delta = wolf
                delta_fitness = fitness

        wolves = update_wolves(wolves, alpha, beta, delta, profits, scout_step_length, alpha_step_length, scout_ratio,
                               update_ratio)

        # 检查是否满足终止条件
        if abs(alpha_fitness - target_fitness) <= tolerance:
            break

    return alpha, alpha_fitness


def has_duplicates(matrix):
    seen = set()
    for row in matrix:
        if row in seen:
            return True
        seen.add(row)
    return False


# 示例参数
num_tasks = 60
num_wolves = 1000
max_iter = 1000
target_fitness = 609.705627  # 目标收益值
tolerance = 0.1  # 允许的误差范围

# 经典狼群算法参数
scout_step_length = 10  # 探狼最大游走步长
alpha_step_length = 5  # 头狼游走步长
scout_ratio = 0.5  # 探狼比例因子
update_ratio = 0.3  # 更新比例因子


A = copy.deepcopy(profit_matrix)

start = time.time()

best_solution, best_fitness = wolf_pack_algorithm(num_wolves, num_tasks, A, max_iter, target_fitness, tolerance,
                                                  scout_step_length, alpha_step_length, scout_ratio, update_ratio)

end = time.time()

print("分配方案中是否有重复元素:", has_duplicates(best_solution))
print("最佳任务分配方案:", best_solution)
print("最佳总收益:", best_fitness)
print("算法运行时间：", (end - start))
print("误差为：", (target_fitness - best_fitness))
