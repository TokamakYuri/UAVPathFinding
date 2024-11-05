import numpy as np
import copy
import time
import matplotlib.pyplot as plt

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
def scout_wolf_walk(wolf, alpha, history_best, num_tasks):
    new_wolf = wolf.copy()
    strategy = np.random.choice([1, 2, 3])  # 随机选择一种策略
    #strategy = np.random.choice([1, 2])  # 随机选择一种策略
    for i in range(num_tasks):
        if strategy == 1:
            # 基于个体知识的探狼游走规则
            new_wolf[i] = int(wolf[i] + np.random.uniform(-1, 1) * (wolf[i] - np.random.choice(wolf)))
        elif strategy == 2:
            # 基于历史知识的探狼游走规则
            new_wolf[i] = int(wolf[i] + np.random.uniform(-1, 1) * (history_best[i] - wolf[i]))
        elif strategy == 3:
            # 基于邻域搜索的探狼游走规则
            e = 0.1  # 很小的正数，用于精细搜索
            new_wolf[i] = int(np.clip(wolf[i] + np.random.uniform(-e, e), 0, num_tasks-1))

    # 修复位置
    return repair_solution(new_wolf)

# 自适应步长的猛狼围攻规则
def adaptive_step_length_attack(wolf, alpha, step, area, t, num_tasks):
    new_wolf = wolf.copy()
    for i in range(num_tasks):
        step_length = step / (1 + area * t)
        new_wolf[i] = int(new_wolf[i] + step_length * (alpha[i] - new_wolf[i]))
    return repair_solution(new_wolf)

# 更新狼的位置
def update_wolves(wolves, alpha, history_best,beta, delta, profits, scout_step_length, alpha_step_length, scout_ratio, update_ratio, step, area, t):
    new_wolves = []
    num_wolves = len(wolves)
    num_tasks = len(alpha)

    for wolf in wolves:
        new_wolf = wolf.copy()
        rand_val = np.random.rand()
        if rand_val < scout_ratio:
            # 探狼随机游走
            new_wolf = scout_wolf_walk(wolf, alpha, history_best, num_tasks)
        elif rand_val < scout_ratio + update_ratio:
            # 头狼根据alpha位置更新
            for i in range(num_tasks):
                new_wolf[i] = int(alpha[i] + np.random.uniform(-alpha_step_length, alpha_step_length))
        else:
            # 猛狼自适应步长围攻
            new_wolf = adaptive_step_length_attack(wolf, alpha, step, area, t, num_tasks)

        # 修复位置
        new_wolf = repair_solution(new_wolf)
        new_wolves.append(new_wolf)
    return new_wolves

def local_search(solution, profits):
    # 局部搜索以优化解的质量
    best_solution = solution.copy()
    best_fitness = calculate_fitness(best_solution, profits)
    num_tasks = len(solution)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            new_solution = solution.copy()
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_fitness = calculate_fitness(new_solution, profits)
            if new_fitness > best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
    return best_solution, best_fitness

def wolf_pack_algorithm(num_wolves, num_tasks, profits, max_iter, target_fitness, tolerance, scout_step_length,
                        alpha_step_length, scout_ratio, update_ratio, step, area):
    # 初始化
    wolves = initialize_wolves(num_wolves, num_tasks)
    alpha = beta = delta = wolves[0]
    alpha_fitness = beta_fitness = delta_fitness = -float('inf')
    history_best = alpha.copy()

    error_history = [] # 记录误差

    for t in range(max_iter):
        for wolf in wolves:
            fitness = calculate_fitness(wolf, profits)  # 计算适应度
            if fitness > alpha_fitness:
                delta, beta, alpha = beta, alpha, wolf
                alpha_fitness = fitness
                history_best = alpha.copy()  # 更新历史最优解
            elif fitness > beta_fitness:
                delta, beta = beta, wolf
                beta_fitness = fitness
            elif fitness > delta_fitness:
                delta = wolf
                delta_fitness = fitness

        wolves = update_wolves(wolves, alpha, beta, history_best, delta, profits, scout_step_length, alpha_step_length, scout_ratio,
                               update_ratio, step, area, t)

        # 局部搜索
        alpha, alpha_fitness = local_search(alpha, profits)  # 头狼自主游走再次搜索

        # 记录当前误差
        current_error = abs(alpha_fitness - target_fitness)
        error_history.append(current_error)

        # 检查是否满足终止条件
        if abs(alpha_fitness - target_fitness) <= tolerance:
            break

    return alpha, alpha_fitness, error_history

# 绘制误差曲线
def plot_error_curve(error_history):
    plt.figure()
    plt.plot(error_history, label='Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

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
tolerance = 0.1 # 允许的误差范围

# 经典狼群算法参数
scout_step_length = 10  # 探狼最大游走步长
alpha_step_length = 5  # 头狼游走步长
scout_ratio = 0.5  # 探狼比例因子
update_ratio = 0.3  # 更新比例因子
step = 10  # 初始猛狼步长
area = 0.1  # 步长衰减率

A = copy.deepcopy(profit_matrix)

start = time.time()

best_solution, best_fitness, error_history = wolf_pack_algorithm(num_wolves, num_tasks, A, max_iter, target_fitness, tolerance,
                                                  scout_step_length, alpha_step_length, scout_ratio, update_ratio,
                                                  step, area)

end = time.time()

print("分配方案中是否有重复元素:", has_duplicates(best_solution))
print("最佳任务分配方案:", best_solution)
print("最佳总收益:", best_fitness)
print("算法运行时间：", (end - start))
print("误差为：", (target_fitness - best_fitness))

# 绘制误差曲线
plot_error_curve(error_history)