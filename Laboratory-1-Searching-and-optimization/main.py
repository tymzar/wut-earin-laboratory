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

bfs_maze = Search(maze, start_position, finish_position, "bfs")
dfs_maze = Search(maze, start_position, finish_position, "dfs")

bfs_final_path, bfs_history, _ = bfs_maze.search()
dfs_final_path, dfs_history, _ = dfs_maze.search()


bfs_steps_amount = len(bfs_final_path) - 1
dfs_steps_amount = len(dfs_final_path) - 1

print(f"BFS: Number of steps in path: {bfs_steps_amount}")
print(f"DFS: Number of steps in path: {dfs_steps_amount}")

print(f"Grid size: {size}")

visualize_data(bfs_history, dfs_history, size, maze, start_position, finish_position, bfs_final_path, dfs_final_path)
