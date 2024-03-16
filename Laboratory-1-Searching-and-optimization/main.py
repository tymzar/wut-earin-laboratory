from maze import Search
from visualize import visualize_data

# Example usage:
maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]

start_position = (0, 0)
finish_position = (0, 2)

current_maze = Search(maze, start_position, finish_position, "bfs")

steps_amount, history, last_node = current_maze.search()

size, step_history = history

print(f"Number of steps in path: {steps_amount}")
print(f"Grid size: {size}")

final_path = current_maze.get_final_path(last_node)

visualize_data(history, size, maze, start_position, finish_position, final_path)
