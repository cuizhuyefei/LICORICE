import re

def concept_str_to_list(text):
    # 定义要匹配的关键词及其顺序
    keywords = [
        "x coordinate of the ball", "y coordinate of the ball", "x velocity of the ball", "y velocity of the ball",
        "velocity of the paddle", "acceleration of the paddle", "jerk of the paddle"
    ]
    
    # 创建一个列表来存储提取的值
    values = [None] * len(keywords)
    
    # 将文本按行分割
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for i, keyword in enumerate(keywords):
            if keyword in line:
                # 使用正则表达式提取数字
                match = re.search(r'-?\d+', line[line.find(keyword)+len(keyword):])
                if match:
                    values[i] = int(match.group())
    
    # 检查是否所有关键词都匹配到了
    if None in values:
        raise ValueError("Not all keywords were matched and extracted successfully.")

    offset = [88, 0, 88, 88, 88, 88*2, 88*4]
    return [values[i]+offset[i] for i in range(7)]
    
if __name__ == '__main__':
    # Example usage
    concept_str = """
To determine the values, let's analyze the frames:

1. **x coordinate of the ball relative to the paddle in the first frame:**
   - The ball is at approximately x = 40.
   - The paddle is at approximately x = 70.
   - Relative x coordinate = 40 - 70 = -30.

2. **y coordinate of the ball in the first frame:**
   - The ball is at approximately y = 42.

3. **x velocity of the ball calculated from the first and second frames:**
   - First frame x = 40, second frame x = 42.
   - x velocity = 42 - 40 = 2.

4. **y velocity of the ball calculated from the first and second frames:**
   - First frame y = 42, second frame y = 44.
   - y velocity = 44 - 42 = 2.

5. **velocity of the paddle calculated from the first and second frames:**
   - First frame x = 70, second frame x = 70.
   - Velocity = 70 - 70 = 0.

6. **acceleration of the paddle calculated from the first three frames:**
   - First frame x = 70, second frame x = 70, third frame x = 70.
   - Velocity = 0 for all frames, so acceleration = 0.

7. **jerk of the paddle calculated from the four frames:**
   - First frame x = 70, second frame x = 70, third frame x = 70, fourth frame x = 70.
   - Acceleration = 0 for all frames, so jerk = 0.

Final values:

x coordinate of the ball: -30
y coordinate of the ball: 42
x velocity of the ball: 2
y velocity of the ball: 2
velocity of the paddle: 0
acceleration of the paddle: 0
jerk of the paddle: 0
"""

    # print(concept_str_to_list(concept_str))
    # exit(0)

    with open(f'./Pong_frames/concepts.txt', 'r') as f:
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

    acc = [0 for i in range(7)]

    # no ensemble
    for i in range(0, 10):
        with open(f'./Pong_0shot_output/gpt4o_output_{i}.txt', 'r') as f:
            concept_str = f.read()
        concept_values = concept_str_to_list(concept_str)
        # concept_values = parse_concept_values_only_position(concept_str)
        concept_gt = lists[i]
        for j in range(7):
            acc[j] += (concept_values[j]-concept_gt[j]) ** 2
        print(i, "prediction:", [x for x in concept_values], "ground truth:", [x for x in concept_gt])
    print([x/10 for x in acc])