import re

def concept_str_to_list(text):
    # 定义要匹配的关键词及其顺序
    keywords = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    # 创建一个空列表来存储提取的值
    values = [None] * len(keywords)  # 使用None初始化，以便于后续检查

    # 将文本按行分割并逆序遍历
    lines = text.split('\n')[::-1]
    for line in lines:
        for i, keyword in enumerate(keywords):
            if keyword in line and values[i] is None:
                # 使用正则表达式提取数字
                match = re.search(r'[-+]?[0-9]*\.?[0-9]+', line)
                if match:
                    values[i] = float(match.group())
    
    # 检查是否所有关键词都匹配到了
    if None in values:
        raise ValueError("Not all keywords were matched and extracted successfully.")

    return values
    
if __name__ == '__main__':
    # Example usage
    concept_str = """
Based on the provided frames, here are the estimated values for the latest frame:

1. **Cart Position**: The cart appears to be moving slightly to the right over the frames. Given the range (-2.4, 2.4), the cart position in the last frame seems to be around **0.1**.

2. **Cart Velocity**: The cart is moving to the right, and the movement seems to be increasing. Therefore, the cart velocity is estimated to be around **0.2**.

3. **Pole Angle**: The pole is tilting to the right, and the tilt appears to be increasing. Given the range (-0.2095, 0.2095), the pole angle in the last frame seems to be around **0.1**.

4. **Pole Angular Velocity**: The pole's tilt is increasing to the right, indicating a positive angular velocity. The pole angular velocity is estimated to be around **0.15**.

So, the estimated values are:

Cart Position: 0.1
Cart Velocity: 0.2
Pole Angle: 0.1
Pole Angular Velocity: 0.15

"""

#     print(concept_str_to_list(concept_str))
#     exit(0)

    with open(f'./CartPole_frames/concepts.txt', 'r') as f:
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

    acc = [0 for i in range(4)]

    # no ensemble
    for i in range(0, 10):
        with open(f'./CartPole_0shot_output/gpt4o_output_{i}.txt', 'r') as f:
            concept_str = f.read()
        concept_values = concept_str_to_list(concept_str)
        # concept_values = parse_concept_values_only_position(concept_str)
        concept_gt = lists[i]
        for j in range(4):
            acc[j] += (concept_values[j]-concept_gt[j]) ** 2
        print(i, "prediction:", [x for x in concept_values], "ground truth:", [x for x in concept_gt])
    print([x/10 for x in acc])