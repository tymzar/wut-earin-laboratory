import unittest
from maze import Search

class TestMaze(unittest.TestCase):
    def test_unreachable(self):
        maze = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        bfs_step, _, _  = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        dfs_step, _, _  = dfs_maze.search()

        self.assertEqual(-1, bfs_step)
        self.assertEqual(-1, dfs_step)

if __name__ == '__main__':
    unittest.main()