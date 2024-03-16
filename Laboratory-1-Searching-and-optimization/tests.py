import unittest
from maze import Search

class TestMaze(unittest.TestCase):
    def test_unreachable(self):
        maze = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        bfs_final_path, _, _  = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        dfs_final_path, _, _  = dfs_maze.search()

        self.assertEqual([], bfs_final_path)
        self.assertEqual([], dfs_final_path)

if __name__ == '__main__':
    unittest.main()