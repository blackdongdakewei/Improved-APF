import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
#这一版解决了局部极小值问题（动态的随机抖动力），解决了斥力过大无法到达目标点的问题（动态K值），解决了多目标规划问题（决策树），但是存在路径抖动的问题(使用路径平滑进行了处理)。
K_att = 15#引力系数
K_rep = 30#斥力初始系数
Q = 5   #避障范围
K_perp = 0.05   #随机力系数
step = 0.1  #步宽（这个步宽乘力是步长）
R = 1.0 #机械臂半径
E = 2   #改进斥力系数
K_distance = 10 #工作空间的数量级
max_step=1  #最大步长
#下一步改进方向，在前往一个目标时忽略其余的目标，遇到局部极小值
def calculate_shortest_path(start, goals):
    path = [start]
    remaining_goals = goals.copy()
    current_point = start
    while remaining_goals:
        distances = [np.linalg.norm(goal - current_point) for goal in remaining_goals]
        closest_goal_index = np.argmin(distances)
        current_point = remaining_goals.pop(closest_goal_index)
        path.append(current_point)
    return np.array(path)


def attractive_potential(goal, position, K_att):
    return 0.5 * K_att * np.sum((goal - position) ** 2)


def repulsive_potential(obstacle_center, obstacle_radius, position, K_rep, Q, goal, K_distance):
    distance = np.linalg.norm(obstacle_center - position) - obstacle_radius
    direction_to_goal = (goal - position) / np.linalg.norm(goal - position)



#这里是改进的重点，将K值变成了动态值，并且采用动态平方（X^x），K越小，排斥力越小。
    GP = np.linalg.norm(goal - position)
    print(position)
    #print(np.linalg.norm(goal - position))
    if np.linalg.norm(goal - position) < K_distance * 1000:
        if (1 / GP) * K_distance >= 1:
            K_rep = K_rep / (1 / (np.linalg.norm(goal - position) / K_distance) ** ((1 / GP) * K_distance))
        elif (1 / GP) * K_distance < 1:
            K_rep = K_rep / (1 / (np.linalg.norm(goal - position) / K_distance) ** 2)
    #print((1 / GP) * K_distance)
    #print(K_rep)




    if distance <= Q:
        force1 = K_rep * ((1 / distance) - (1 / Q)) * (1 / distance ** 2)
        force2 = 0.5 * K_rep * ((1 / distance) - (1 / Q)) ** 2 * (1 - np.exp(-np.sum((goal - position) ** 2) / R ** 2))
        return force1 * (position - obstacle_center) / np.linalg.norm(
            position - obstacle_center) + force2 * direction_to_goal
    else:
        return np.zeros(3)


def total_potential_and_force(goal, obstacles, position, K_att, K_rep, Q, K_perp, K_distance):
    U_att = attractive_potential(goal, position, K_att)
    U_rep = sum(
        repulsive_potential(obstacle['center'], obstacle['radius'], position, K_rep, Q, goal, K_distance) for obstacle
        in obstacles)
    potential = U_att + U_rep

    attractive_force = K_att * (goal - position)
    repulsive_forces = sum(
        repulsive_potential(obstacle['center'], obstacle['radius'], position, K_rep, Q, goal, K_distance) for obstacle
        in obstacles)

    total_force = attractive_force + repulsive_forces
    return potential, total_force


def dynamic_obstacle_avoidance_path(initial_path, obstacles, K_att, K_rep, Q, K_perp, K_distance,max_step):
    dynamic_path = [initial_path[0]]
    position = initial_path[0]
    for goal in initial_path[1:]:
        while np.linalg.norm(goal - position) >= 0.1:
            _, force = total_potential_and_force(goal, obstacles, position, K_att, K_rep, Q, K_perp, K_distance)
            step_force = step * force
            print('step_force:',np.linalg.norm(step_force))
            # 限制每一步走过的最大距离为0.5
            if np.linalg.norm(step_force) > max_step:
                step_force = max_step * step_force / np.linalg.norm(step_force)

            position = position + step_force
            dynamic_path.append(position)
            ax.plot([dynamic_path[-2][0], dynamic_path[-1][0]], [dynamic_path[-2][1], dynamic_path[-1][1]],
                    [dynamic_path[-2][2], dynamic_path[-1][2]], color='g')
            plt.pause(0.01)

        dynamic_path.append(goal)

    return np.array(dynamic_path)


def remove_zigzag(path, threshold=0.1):
    new_path = [path[0]]
    for i in range(1, len(path) - 1):
        direction1 = path[i] - path[i - 1]
        direction2 = path[i + 1] - path[i]
        angle = np.dot(direction1, direction2) / (np.linalg.norm(direction1) * np.linalg.norm(direction2))
        if angle > threshold:
            new_path.append(path[i])
    new_path.append(path[-1])
    return np.array(new_path)


def smooth_path(path, threshold=0.1):
    straight_segments = [path[0]]
    smoothed_path = []
    for i in range(1, len(path) - 1):
        direction1 = path[i] - path[i - 1]
        direction2 = path[i + 1] - path[i]
        angle = np.dot(direction1, direction2) / (np.linalg.norm(direction1) * np.linalg.norm(direction2))
        if angle < threshold:
            straight_segments.append(path[i])
        else:
            if len(straight_segments) > 3:
                tck, u = splprep([np.array(straight_segments)[:, 0], np.array(straight_segments)[:, 1],
                                  np.array(straight_segments)[:, 2]], s=2)
                u_new = np.linspace(0, 1, len(straight_segments))
                x_new, y_new, z_new = splev(u_new, tck)
                smoothed_path.extend(np.vstack((x_new, y_new, z_new)).T)
            else:
                smoothed_path.extend(straight_segments)
            smoothed_path.append(path[i])
            straight_segments = [path[i]]
    smoothed_path.extend(straight_segments)
    return np.array(smoothed_path)
#设置障碍点，目标点，起始点
goals = [np.array([5, 3, 5]), np.array([4, 8, 1]), np.array([10, 10, 10])]
obstacles = [{'center': np.array([5, 5, 5]), 'radius': 1},
             {'center': np.array([7, 7, 7]), 'radius': 0.8},
             {'center': np.array([7, 7, 2]), 'radius': 2},
             {'center': np.array([4, 8, 10]), 'radius': 1},
             {'center': np.array([10, 8, 4]), 'radius': 2},
             {'center': np.array([8, 4, 10]), 'radius': 0.5},
             {'center': np.array([10, 4, 8]), 'radius': 1},
             {'center': np.array([4, 10, 8]), 'radius': 4},
             {'center': np.array([8, 10, 4]), 'radius': 3},
             {'center': np.array([1, 10, 5]), 'radius': 3},
             {'center': np.array([2, 4, 3]), 'radius': 3},
             {'center': np.array([2, 2, 2]), 'radius': 1},
             {'center': np.array([2, 1, 2]), 'radius': 0.5},
             {'center': np.array([2, 6, 2]), 'radius': 0.5},
{'center': np.array([1, 9, 9]), 'radius': 0.2},
{'center': np.array([9, 9, 1]), 'radius': 0.5},
{'center': np.array([4,6,2]), 'radius': 2},
{'center': np.array([9,9,9]), 'radius': 1.5},

             ]

start = np.array([0, 0, 0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(goals[-1][0], goals[-1][1], goals[-1][2], color='r', marker='x', s=100, label='Goal')
ax.scatter(goals[-2][0], goals[-2][1], goals[-2][2], color='r', marker='x', s=100, label='Goal')
ax.scatter(goals[-3][0], goals[-3][1], goals[-3][2], color='r', marker='x', s=100, label='Goal')
for obstacle in obstacles:
    ax.scatter(obstacle['center'][0], obstacle['center'][1], obstacle['center'][2], color='b',
               s=100 * obstacle['radius'], alpha=0.5, label='Obstacle')

# 计算最短路径
initial_path = calculate_shortest_path(start, goals)
# 动态避障
final_path = dynamic_obstacle_avoidance_path(initial_path, obstacles, K_att, K_rep, Q, K_perp, K_distance,max_step)
# 去除锯齿状曲线
path = remove_zigzag(np.array(final_path))

# 平滑路径
path = smooth_path(path)
# 绘制路径
ax.plot(final_path[:, 0], final_path[:, 1], final_path[:, 2], color='g', label='Path')
ax.plot(path[:, 0], path[:, 1], path[:, 2], color='b', label='Smooth Path')
#ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Dynamic Obstacle Avoidance Path')
plt.show()