import re

def concept_str_to_list(concept_str):
    # 使用正则表达式提取 agent 位置
    agent_position = tuple(map(int, re.search(r'Agent Position: \((\d+), (\d+)\)', concept_str).groups()))
    # 使用正则表达式提取 agent 方向
    agent_direction = re.search(r'Agent Direction: (\w+)', concept_str).group(1)
    
    # 匹配障碍物位置
    first_obstacle = tuple(map(int, re.search(r'First Obstacle: \((\d+), (\d+)\)', concept_str).groups()))
    second_obstacle = tuple(map(int, re.search(r'Second Obstacle: \((\d+), (\d+)\)', concept_str).groups()))
    third_obstacle_match = re.search(r'Third Obstacle: \((\d+), (\d+)\)', concept_str)

    obstacles = [first_obstacle, second_obstacle]
    if third_obstacle_match:
        third_obstacle = tuple(map(int, third_obstacle_match.groups()))
        obstacles.append(third_obstacle)
    
    # 按照 x 值和 y 值排序
    obstacles.sort(key=lambda x: (x[0], x[1]))

    # 使用正则表达式提取方向的布尔值
    right = re.search(r'Right.*: (\w+)', concept_str).group(1) == 'true'
    down = re.search(r'Down.*: (\w+)', concept_str).group(1) == 'true'
    left = re.search(r'Left.*: (\w+)', concept_str).group(1) == 'true'
    up = re.search(r'Up.*: (\w+)', concept_str).group(1) == 'true'

    # 映射方向
    direction_map = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
    agent_direction_value = direction_map[agent_direction]

    # 将布尔值转换为整数
    right_value = int(right)
    down_value = int(down)
    left_value = int(left)
    up_value = int(up)

    # 拼接成整数列表
    result_list = [
        agent_position[0], agent_position[1], 
        agent_direction_value
    ]
    
    # 添加障碍物坐标
    for obstacle in obstacles:
        result_list.extend([obstacle[0], obstacle[1]])
    
    # 添加方向的布尔值
    result_list.extend([right_value, down_value, left_value, up_value])

    return result_list

def extract_integers(s):
    numbers = re.findall(r'\d+', s)  # 使用正则表达式找到所有连续的数字
    return list(map(int, numbers))   # 将找到的数字字符串转换成整数列表

def parse_concept_values_only_position(concept_str):
    x = extract_integers(concept_str)
    while len(x) < 14:
        x.append(-1)
    return x
    
if __name__ == '__main__':
    concept_str1 = '''
    1. Agent Position: (1, 1)
    2. Agent Direction: down
    3. First Obstacle: (2, 1)
    4. Second Obstacle: (2, 2)
    5. Right (x + 1): false
    6. Down (y + 1): true
    7. Left (x - 1): false
    8. Up (y - 1): false
    '''

    concept_str2 = '''
    1. Agent Position: (1, 2)
    2. Agent Direction: left
    3. Obstacle Position:
    - First Obstacle: (2, 2)
    - Second Obstacle: (2, 3)
    4. Direction Movable:
    - Right (x + 1): false
    - Down (y + 1): true
    - Left (x - 1): false
    - Up (y - 1): true
    '''

    concept_str3 = '''
    1. Agent Position: (1, 2)
    2. Agent Direction: right
    3. Obstacle Position:
    - First Obstacle: (3, 3)
    - Second Obstacle: (2, 3)
    - Third Obstacle: (2, 1)
    4. Direction Movable:
    - Right (x + 1): true
    - Down (y + 1): true
    - Left (x - 1): false
    - Up (y - 1): true
    '''

    concept_str4 = '''
    1. Agent Position: (1, 2)
    2. Agent Direction: right
    3. Obstacle Position:
    - First Obstacle: (2, 1)
    - Second Obstacle: (2, 3)
    4. Direction Movable:
    - Right (x + 1): true
    - Down (y + 1): true
    - Left (x - 1): false
    - Up (y - 1): true
    '''

    # print(concept_str_to_list(concept_str1))
    # print(concept_str_to_list(concept_str2))
    # print(concept_str_to_list(concept_str3))
    # print(concept_str_to_list(concept_str4))
    # exit(0)

    with open(f'./DynamicObstacles_v3_large_frames_160/concepts.txt', 'r') as f:
        lines = f.read().strip().split('\n')

    lists = []

    # 遍历每一行
    for line in lines:
        # 去掉行首行尾的空白字符
        line = line.strip()
        # 找到方括号内的部分，并去除冒号和空白字符
        numbers_str = line.split('[')[-1].strip(']')
        # 将字符串分割成单个数字字符串，去除点并转换成float类型
        numbers = [float(num.strip('.')) for num in numbers_str.split()]
        # 将得到的数字列表添加到lists中
        lists.append(numbers)

    # print(lists)

    N = len(lists[0]) # 11 or 13
    acc = [0 for i in range(N)]

    # no ensemble
    for i in range(0, 10):
        with open(f'./DynamicObstacles_v3_large_0shot_output/gpt4o_output_{i}.txt', 'r') as f:
            concept_str = f.read()
        concept_values = concept_str_to_list(concept_str)
        # concept_values = parse_concept_values_only_position(concept_str)
        concept_gt = lists[i]
        for j in range(N):
            if concept_values[j] == concept_gt[j]:
                acc[j] += 1
        print(i, "prediction:", [int(x) for x in concept_values], "ground truth:", [int(x) for x in concept_gt])
    print([x/10 for x in acc])

# have ensemble
# from statistics import mode, StatisticsError
# def calculate_mode_across_lists(lists):
#     # 转置列表，以便每个新的子列表包含所有原列表相同位置的元素
#     transposed_lists = list(zip(*lists))
#     mode_list = []

#     for idx, position_list in enumerate(transposed_lists):
#         if idx >= 5:
#             mode_value = sum(position_list) > 0
#         else:
#             mode_value = mode(position_list)
#         mode_list.append(mode_value)
#     return mode_list

# for i in range(20):
#     concept_values_ensembled = []
#     for t in range(5):
#         with open(f'./DynamicObstacles_0shot_output_ensemble/gpt4o_output_{i}_{t}.txt', 'r') as f:
#             concept_str = f.read()
#         concept_values = concept_str_to_list(concept_str)
#         concept_values_ensembled.append(concept_values)
#     concept_values = calculate_mode_across_lists(concept_values_ensembled)
#     concept_gt = lists[i]
#     for j in range(5):
#         if concept_values[j] == concept_gt[j]:
#             acc[j] += 1
#     print(i, "prediction:", [int(x) for x in concept_values], "ground truth:", [int(x) for x in concept_gt])
# print([x/20 for x in acc])
