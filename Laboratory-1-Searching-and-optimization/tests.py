import unittest
from maze import Search

class TestMaze(unittest.TestCase):
    def test_unreachable(self):
        maze = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        current_maze = Search(maze, start_position, finish_position, "bfs")
        step_number, history, last_node = current_maze.search()
        self.assertEqual(-1, step_number)

if __name__ == '__main__':
    unittest.main()