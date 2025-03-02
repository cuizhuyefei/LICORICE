import re

def concept_str_to_list(text):
    # 定义要匹配的关键词及其顺序
    keywords = [
        "frame 1 white x", "frame 1 white y", "frame 1 black x", "frame 1 black y",
        "frame 2 white x", "frame 2 white y", "frame 2 black x", "frame 2 black y"
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

    return values
    
if __name__ == '__main__':
    # Example usage
    concept_str = """
frame 1 white x: 50
frame 1 white y: 100
frame 1 black x: 90
frame 1 black y: 100
frame 2 white x: 50
frame 2 white y: 110
frame 2 black x: 90
frame 2 black y: 110
"""

#     print(concept_str_to_list(concept_str))
#     exit(0)

    with open(f'./Boxing_frames/concepts.txt', 'r') as f:
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

    acc = [0 for i in range(8)]

    # no ensemble
    for i in range(0, 5):
        with open(f'./Boxing_0shot_output/gpt4o_output_{i}.txt', 'r') as f:
            concept_str = f.read()
        concept_values = concept_str_to_list(concept_str)
        # concept_values = parse_concept_values_only_position(concept_str)
        concept_gt = lists[i]
        for j in range(8):
            acc[j] += (concept_values[j]-concept_gt[j]) ** 2
        print(i, "prediction:", [x for x in concept_values], "ground truth:", [x for x in concept_gt])
    print([x/10 for x in acc])