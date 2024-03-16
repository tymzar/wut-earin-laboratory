from maze import Search
from visualize import visualize_data
from maze import generate_maze, random_coordinate
import numpy

# Example usage:
size = (10, 10)
maze = generate_maze(size)
#maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]

print(maze)

start_position = random_coordinate(size)
finish_position = random_coordinate(size)

print(start_position)
print(finish_position)

current_maze = Search(maze, start_position, finish_position, "bfs")

steps_amount, history, last_node = current_maze.search()

size, step_history = history

print(f"Number of steps in path: {steps_amount}")
print(f"Grid size: {size}")

final_path = current_maze.get_final_path(last_node)

visualize_data(history, size, maze, start_position, finish_position, final_path)
