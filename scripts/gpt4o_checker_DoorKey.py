import re

def extract_integers(s):
    numbers = re.findall(r'\d+', s)  # 使用正则表达式找到所有连续的数字
    return list(map(int, numbers))   # 将找到的数字字符串转换成整数列表

def parse_concept_values_only_position(concept_str):
    x = extract_integers(concept_str)
    while len(x) < 12:
        x.append(-1)
    return x

def concept_str_to_list(concept_str):
    # 使用正则表达式提取各个值
    agent_position = tuple(map(int, re.search(r'Agent Position: \((\d+), (\d+)\)', concept_str).groups()))
    agent_direction = re.search(r'Agent Direction: (\w+)', concept_str).group(1)
    key_position = tuple(map(int, re.search(r'Key Position: \((\d+), (\d+)\)', concept_str).groups()))
    # 匹配门的位置和状态
    door_position_match = re.search(r'Door Position: \((\d+), (\d+)\)', concept_str)
    door_status_match = re.search(r'Door Status: (\w+)', concept_str)
    if door_position_match and door_status_match:
        door_position = tuple(map(int, door_position_match.groups()))
        door_status = door_status_match.group(1) == 'true'
    else:
        door_position = tuple(map(int, re.search(r'Door Position:\s*- Position: \((\d+), (\d+)\)', concept_str).groups()))
        door_status = re.search(r'- Status: (\w+)', concept_str).group(1) == 'true'
    # 使用正则表达式提取方向的布尔值
    right = re.search(r'- Right.*: (\w+)', concept_str).group(1) == 'true'
    down = re.search(r'- Down.*: (\w+)', concept_str).group(1) == 'true'
    left = re.search(r'- Left.*: (\w+)', concept_str).group(1) == 'true'
    up = re.search(r'- Up.*: (\w+)', concept_str).group(1) == 'true'

    # 映射方向
    direction_map = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
    agent_direction_value = direction_map[agent_direction]

    # 将布尔值转换为整数
    door_status_value = int(door_status)
    right_value = int(right)
    down_value = int(down)
    left_value = int(left)
    up_value = int(up)

    # 拼接成整数列表
    result_list = [
        agent_position[0], agent_position[1], 
        agent_direction_value, 
        key_position[0], key_position[1], 
        door_position[0], door_position[1], 
        door_status_value, 
        right_value, down_value, left_value, up_value
    ]

    return result_list

if __name__ == '__main__':
    # 测试函数
    concept_str = """
1. Agent Position: (1, 4)
2. Agent Direction: up
3. Key Position: (2, 1)
4. Door Position:
   - Position: (2, 3)
   - Status: false
5. Direction Movable:
   - Right (x + 1): false
   - Down (y + 1): false
   - Left (x - 1): false
   - Up (y - 1): true
"""

    print(concept_str_to_list(concept_str))
    exit(0)

    with open(f'./DoorKey_frames_160/concepts.txt', 'r') as f:
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

    acc = [0 for i in range(12)]

    # no ensemble
    for i in range(0, 50):
        with open(f'./DoorKey_0shot_output_new/gpt4o_output_{i}.txt', 'r') as f:
            concept_str = f.read()
        concept_values = concept_str_to_list(concept_str)
        # concept_values = parse_concept_values_only_position(concept_str)
        concept_gt = lists[i]
        for j in range(12):
            if concept_values[j] == concept_gt[j]:
                acc[j] += 1
        print(i, "prediction:", [int(x) for x in concept_values], "ground truth:", [int(x) for x in concept_gt])
    print([x/50 for x in acc])
    print(sum(acc)/600)

# have ensemble
# from statistics import mode, StatisticsError
# def calculate_mode_across_lists(lists):
#     # 转置列表，以便每个新的子列表包含所有原列表相同位置的元素
#     transposed_lists = list(zip(*lists))
#     mode_list = []

#     for idx, position_list in enumerate(transposed_lists):
#         mode_value = mode(position_list)
#         mode_list.append(mode_value)
#     return mode_list

# for i in range(20):
#     concept_values_ensembled = []
#     for t in range(5):
#         with open(f'./DoorKey_0shot_output_ensemble/gpt4o_output_{i}_{t}.txt', 'r') as f:
#             concept_str = f.read()
#         concept_values = concept_str_to_list(concept_str)
#         concept_values_ensembled.append(concept_values)
#     concept_values = calculate_mode_across_lists(concept_values_ensembled)
#     concept_gt = lists[i]
#     for j in range(12):
#         if concept_values[j] == concept_gt[j]:
#             acc[j] += 1
#     print(i, "prediction:", [int(x) for x in concept_values], "ground truth:", [int(x) for x in concept_gt])
# print([x/20 for x in acc])

# acc = [0 for i in range(12)]
# for t in range(5):
#     for i in range(20):
#         concept_values_ensembled = []
#         with open(f'./DoorKey_0shot_output_ensemble/gpt4o_output_{i}_{t}.txt', 'r') as f:
#             concept_str = f.read()
#         concept_values = concept_str_to_list(concept_str)
#         concept_values_ensembled.append(concept_values)
#         concept_values = calculate_mode_across_lists(concept_values_ensembled)
#         concept_gt = lists[i]
#         for j in range(12):
#             if concept_values[j] == concept_gt[j]:
#                 acc[j] += 1
# print([x/100 for x in acc])
