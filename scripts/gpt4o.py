import os, cv2
import openai
from openai import OpenAI
import base64
openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

prompt_cart_pole = """Here are the past 4 rendered frames from the CartPole environment. Please use these images to estimate the following values in the latest frame (the last one):

- Cart Position, within the range (-2.4, 2.4)
- Cart Velocity
- Pole Angle, within the range (-0.2095, 0.2095)
- Pole Angular Velocity
Additionally, please note that the last action taken was [_].

Please carefully determine the following values and give concise answers one by one. Make sure to return an estimated value for each parameter, even if the task may look challening.

Follow the reporting format:

Cart Position: estimated_value
Cart Velocity: estimated_value
Pole Angle: estimated_value
Pole Angular Velocity: estimated_value
"""

prompt_dynamic_obstacles = """Here is an image of a 3x3 grid composed of black cells, with each cell either empty or containing an object. Within this grid, there is a red isosceles triangle representing the agent, two blue balls representing obstacles, and one green square representing the goal.
Please carefully determine the following values and give concise answers one by one:
- agent_at_right: boolean, true if the agent is in the rightmost column (which is also directly above the green square).
- agent_at_bottom: boolean, true if the agent is at the bottom row (which is also directly to the left of the green square).
- agent_direction: choose from 'right', 'down', 'left', or 'up', describing the orientation of the agent (the direction of its vertex points).
- obstacle_right: boolean, true if the cell immediately to the right of the agent contains a blue ball.
- obstacle_below: boolean, true if the cell immediately below the agent contains a blue ball.
"""

prompt_dynamic_obstacles_v2 = """Here is an image of a 3x3 grid composed of black cells, with each cell either empty or containing an object. Within this grid, there is a red isosceles triangle representing the agent, two blue balls representing obstacles, and one green square representing the goal.
Please carefully determine the 9 following values and give concise answers one by one:
1. Top-left cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the top-left cell.
2. Middle-left cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the middle-left cell.
3. Bottom-left cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the bottom-left cell.
4. Top-middle cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the top-middle cell.
5. Middle-middle cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the middle-middle cell.
6. Bottom-middle cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the bottom-middle cell.
7. Top-right cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the top-right cell.
8. Middle-right cell: Choose from 'empty', 'obstacle', or 'agent', describing the object in the middle-right cell.
9. Agent direction: Choose from 'right', 'down', 'left', or 'up', describing the orientation of the agent (the direction of its vertex points).
"""

prompt_dynamic_obstacles_v3 = """Here is an image of a 3x3 grid composed of black cells, with each cell either empty or containing an object. Each cell is defined by an integer-valued coordinate system starting at (1, 1) for the top-left cell. The coordinates increase rightward along the x-axis and downward along the y-axis. Within this grid, there is a red isosceles triangle representing the agent, two blue balls representing obstacles, and one green square representing the goal.
Please carefully determine the following values and give concise answers one by one:
1. Agent Position: Identify and report the coordinates (x, y) of the red triangle (agent). Ensure the accuracy by double-checking the agent's exact location within the grid.
2. Agent Direction: Specify the direction the red triangle is facing, which is the orientation of the vertex (pointy corner) of the isosceles triangle. Choose from 'right', 'down', 'left', or 'up'. Clarify that this direction is independent of movement options.
3. Obstacle Position: Identify and report the coordinates of the two obstacles in ascending order. Compare the coordinates by their x-values first. If the x-values are equal, compare by their y-values.
- First Obstacle: Provide the coordinates (x, y) of the first blue ball.
- Second Obstacle: Provide the coordinates (x, y) of the second blue ball.
4. Direction Movable: Evaluate and report whether the agent can move one cell in each specified direction, namely, the neighboring cell in that direction is active and empty (not obstacle or out of bounds):
- Right (x + 1): Check the cell to the right.
- Down (y + 1): Check the cell below.
- Left (x - 1): Check the cell to the left.
- Up (y - 1): Check the cell above.
Each direction's feasibility should be reported as 'true' if clear and within the grid, and 'false' otherwise.

Reporting Format:
Carefully report each piece of information sequentially, following the format 'name: answer'. Ensure each response is precise and reflects careful verification of the grid details as viewed. 
"""

prompt_dynamic_obstacles_v3_large = """Here is an image of a 4x4 grid composed of black cells, with each cell either empty or containing an object. Each cell is defined by an integer-valued coordinate system starting at (1, 1) for the top-left cell. The coordinates increase rightward along the x-axis and downward along the y-axis. Within this grid, there is a red isosceles triangle representing the agent, three blue balls representing obstacles, and one green square representing the goal.
Please carefully determine the following values and give concise answers one by one:
1. Agent Position: Identify and report the coordinates (x, y) of the red triangle (agent). Ensure the accuracy by double-checking the agent's exact location within the grid.
2. Agent Direction: Specify the direction the red triangle is facing, which is the orientation of the vertex (pointy corner) of the isosceles triangle. Choose from 'right', 'down', 'left', or 'up'. Clarify that this direction is independent of movement options.
3. Obstacle Position: Identify and report the coordinates of the three obstacles in ascending order. Compare the coordinates by their x-values first. If the x-values are equal, compare by their y-values.
- First Obstacle: Provide the coordinates (x, y) of the first blue ball.
- Second Obstacle: Provide the coordinates (x, y) of the second blue ball.
- Third Obstacle: Provide the coordinates (x, y) of the third blue ball.
4. Direction Movable: Evaluate and report whether the agent can move one cell in each specified direction, namely, the neighboring cell in that direction is active and empty (not obstacle or out of bounds):
- Right (x + 1): Check the cell to the right.
- Down (y + 1): Check the cell below.
- Left (x - 1): Check the cell to the left.
- Up (y - 1): Check the cell above.
Each direction's feasibility should be reported as 'true' if clear and within the grid, and 'false' otherwise.

Reporting Format:
Carefully report each piece of information sequentially, following the format 'name: answer'. Ensure each response is precise and reflects careful verification of the grid details as viewed. 
"""

prompt_door_key = """Here is an image of a 4x4 grid composed of black cells, with each cell either empty or containing an object. Each cell is defined by an integer-valued coordinate system starting at (1, 1) for the top-left cell. The coordinates increase rightward along the x-axis and downward along the y-axis. Within this grid, there is a red isosceles triangle representing the agent, a yellow cell representing the door (which may visually disappear if the door is open), a yellow key icon representing the key (which may disappear), and one green square representing the goal. Carefully analyze the grid and report on the following attributes, focusing only on the black cells as the gray cells are excluded from the active black area. 

Detailed Instructions:
1. Agent Position: Identify and report the coordinates (x, y) of the red triangle (agent). Ensure the accuracy by double-checking the agent's exact location within the grid.
2. Agent Direction: Specify the direction the red triangle is facing, which is the orientation of the vertex (pointy corner) of the isosceles triangle. Choose from 'right', 'down', 'left', or 'up'. Clarify that this direction is independent of movement options.
3. Key Position: Provide the coordinates (x, y) where the key is located. If the key is absent, report as (0, 0). Verify visually that the key is present or not before reporting.
4. Door Position:
- Position: Determine and report the coordinates (x, y) of the door.
- Status: Assess whether the door is open or closed (closed means the door is visible as a whole yellow cell, while open means the door disappears visually). Report as 'true' for open and 'false' for closed. Double-check the door's appearance to confirm if it is open or closed.
5. Direction Movable: Evaluate and report whether the agent can move one cell in each specified direction, namely, the neighboring cell in that direction is active and empty (not key, closed door, or grey inactive cell):
- Right (x + 1): Check the cell to the right.
- Down (y + 1): Check the cell below.
- Left (x - 1): Check the cell to the left.
- Up (y - 1): Check the cell above.
Each direction's feasibility should be reported as 'true' if clear and within the grid, and 'false' otherwise.

Reporting Format:
Carefully report each piece of information sequentially, following the format 'name: answer'. Ensure each response is precise and reflects careful verification of the grid details as viewed. """

prompt_door_key_large = """Here is an image of a 5x5 grid composed of black cells, with each cell either empty or containing an object. Each cell is defined by an integer-valued coordinate system starting at (1, 1) for the top-left cell. The coordinates increase rightward along the x-axis and downward along the y-axis. Within this grid, there is a red isosceles triangle representing the agent, a yellow cell representing the door (which may visually disappear if the door is open), a yellow key icon representing the key (which may disappear), and one green square representing the goal. Carefully analyze the grid and report on the following attributes, focusing only on the black cells as the gray cells are excluded from the active black area. 

Detailed Instructions:
1. Agent Position: Identify and report the coordinates (x, y) of the red triangle (agent). Ensure the accuracy by double-checking the agent's exact location within the grid.
2. Agent Direction: Specify the direction the red triangle is facing, which is the orientation of the vertex (pointy corner) of the isosceles triangle. Choose from 'right', 'down', 'left', or 'up'. Clarify that this direction is independent of movement options.
3. Key Position: Provide the coordinates (x, y) where the key is located. If the key is absent, report as (0, 0). Verify visually that the key is present or not before reporting.
4. Door Position:
- Position: Determine and report the coordinates (x, y) of the door.
- Status: Assess whether the door is open or closed (closed means the door is visible as a whole yellow cell, while open means the door disappears visually). Report as 'true' for open and 'false' for closed. Double-check the door's appearance to confirm if it is open or closed.
5. Direction Movable: Evaluate and report whether the agent can move one cell in each specified direction, namely, the neighboring cell in that direction is active and empty (not key, closed door, or grey inactive cell):
- Right (x + 1): Check the cell to the right.
- Down (y + 1): Check the cell below.
- Left (x - 1): Check the cell to the left.
- Up (y - 1): Check the cell above.
Each direction's feasibility should be reported as 'true' if clear and within the grid, and 'false' otherwise.

Reporting Format:
Carefully report each piece of information sequentially, following the format 'name: answer'. Ensure each response is precise and reflects careful verification of the grid details as viewed. """

prompt_boxing = """Here are two consecutive rendered frames from the Atari Boxing environment. The game screen is 160x210 pixels, with (0, 0) at the top-left corner. The x-coordinate increases rightward, and the y-coordinate increases downward.
For each frame, estimate the following values as integers:

- The white player's x and y coordinates
- The black player's x and y coordinates

Please carefully determine the following values and give concise answers one by one. Make sure to return an estimated value for each one, even if the task may look challenging.

Follow the reporting format:

frame 1 white x: estimated_value
frame 1 white y: estimated_value
frame 1 black x: estimated_value
frame 1 black y: estimated_value
frame 2 white x: estimated_value
frame 2 white y: estimated_value
frame 2 black x: estimated_value
frame 2 black y: estimated_value
"""

prompt_pong = """Here are four consecutive rendered frames from the Atari Pong environment. The game screen is 84x84 pixels, with (0, 0) at the top-left corner. You need to look at the ball and the paddle in the right. The x-coordinate increases downward, and the y-coordinate increases rightward.

Please determine the following values and give concise answers one by one. Make sure to return an estimated value for each one, even if the task may look challenging.

- x coordinate of the ball relative to the paddle in the first frame
- y coordinate of the ball in the first frame
- x velocity of the ball calculated from the first and second frames
- y velocity of the ball calculated from the first and second frames
- velocity of the paddle calculated from the first and second frames
- acceleration of the paddle calculated from the first three frames
- jerk of the paddle calculated from the four frames

Follow the reporting format in your response. 
x coordinate of the ball: estimated_value
y coordinate of the ball: estimated_value
x velocity of the ball: estimated_value
y velocity of the ball: estimated_value
velocity of the paddle: estimated_value
acceleration of the paddle: estimated_value
jerk of the paddle: estimated_value
"""


seeds = [42, 123, 456, 789, 10007]

def gpt4o(filename, folder, ensemble_idx=None):
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": prompt_pong,
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encode_image(filename)}",
            },
          },
          # {
          #   "type": "text",
          #   "text": """For the last image provided:""",
          # },
        ],
      }
    ],
    max_tokens=900,
    temperature=0,
    seed=42 if ensemble_idx==None else seeds[ensemble_idx],
  )
  print(response.choices[0].message.content, response.system_fingerprint)
  txt_filename = filename.split('/')[-1].replace('.jpg', '.txt')
  if ensemble_idx != None:
    txt_filename = f'{txt_filename.split(".")[0]}_{ensemble_idx}.txt'
  ensemble_suffix = '_ensemble' if ensemble_idx!=None else ''
  # print(txt_filename)
  # 写入文件
  with open(f'./Pong_0shot_output{ensemble_suffix}/gpt4o_output_{txt_filename}', 'w') as f:
    f.write(response.choices[0].message.content + '\n')
  print(response.usage)

def numerical_sort(value):
  return int(value.split('.')[0])

if __name__ == '__main__':
  folder = './Pong_frames/'
  ensemble = False

  # 遍历folder文件夹下的所有图片
  for filename in [str(idx) + '.jpg' for idx in range(10)]:
    print(filename)
    if ensemble:
      for idx in range(len(seeds)):
        gpt4o(folder + filename, folder, ensemble_idx=idx)
    else:
      gpt4o(folder + filename, folder)

# gpt4o(folder + '55.jpg', folder)
# gpt4o(folder + '59.jpg', folder)
# gpt4o(folder + '43.jpg', folder)
# gpt4o(folder + '37.jpg', folder)
# gpt4o(folder + '32.jpg', folder)
# gpt4o(folder + '51.jpg', folder)
# gpt4o(folder + '40.jpg', folder)
# gpt4o(folder + '47.jpg', folder)
# gpt4o(folder + '49.jpg', folder)
# gpt4o(folder + '53.jpg', folder)
# gpt4o(folder + '57.jpg', folder)

# 27 20 28 26 25 24 22 in the prompt
# 29 agent position, agent direction, door position, direction movable wrong
# 23 agent position, door status, direction movable wrong
# 21 correct!
# 0 agent position, agent direction, key position, direction movable wrong


# [27, 33, 39, 28, 29, 25, 38, 31, 21, 44, 50, 54, 58, 46, 56, 59, 43, 37, 51, 47]

# 59 (WA: 0, 3)
# 43 (WA: 0, 1)
# 37 (WA: 3)
# 51 (WA: 3, 4)
# 47 (WA: 3, 4)
# 57 (WA: 3)
# 55, 32, 40, 49, 53 (AC)


# system_fingerprint fp_927397958d