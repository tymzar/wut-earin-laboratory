import unittest
from maze import Search, generate_maze
from colored import Fore, Style
from main import EscapeSequenceColors
from visualize import visualize_data

class TestMaze(unittest.TestCase):
    def test_unreachable(self):
        maze = [[0, 1, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        bfs_path_steps, bfs_path, _, _ = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        dfs_path_steps, dfs_path, _, _ = dfs_maze.search()

        self.assertEqual(-1, bfs_path_steps)
        self.assertEqual(-1, dfs_path_steps)
        self.assertEqual([], bfs_path)
        self.assertEqual([], dfs_path)

    def test_reachable(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        bfs_path_steps, bfs_path, _, _ = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        dfs_path_steps, dfs_path, _, _ = dfs_maze.search()

        self.assertNotEqual(0, len(dfs_path))
        self.assertNotEqual(0, len(bfs_path))
        self.assertNotEqual(-1, dfs_path_steps)
        self.assertNotEqual(-1, bfs_path_steps)

    def test_invalid_search_type(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "invalid_search_type")

    def test_empty_maze(self):
        maze = []
        start_position = (0, 0)
        finish_position = (1, 2)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_invalid_start_position(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 3)
        finish_position = (1, 2)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_invalid_finish_position(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 3)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_start_position_is_wall(self):
        maze = [[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_finish_position_is_wall(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (0, 1)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_finish_position_is_start_position(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]]
        start_position = (0, 0)
        finish_position = (0, 0)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_generate_maze(self):
        maze, start_position, finish_position = generate_maze((4, 4))
        self.assertEqual(4, len(maze))
        self.assertEqual(4, len(maze[0]))
        self.assertEqual(0, maze[start_position[0]][start_position[1]])
        self.assertEqual(0, maze[finish_position[0]][finish_position[1]])

    def test_irregular_maze(self):
        maze = [[0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1], [1, 1, 1]]
        start_position = (0, 0)
        finish_position = (1, 2)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_invalid_cell_value(self):
        maze = [[0, 1, 0], [3, 1, 0], [0, 0, 0], [0, 1, 0]]
        start_position = (0, 0)
        finish_position = (1, 2)
        with self.assertRaises(ValueError):
            Search(maze, start_position, finish_position, "bfs")

    def test_bfs_dfs_difference_path_length(self):
        maze = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        start_position = (4, 4)
        finish_position = (0, 0)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        _, bfs_final_path, bfs_history, _ = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        _, dfs_final_path, dfs_history, _ = dfs_maze.search()

        visualize_data(
            bfs_history,
            dfs_history,
            maze,
            start_position,
            finish_position,
            bfs_final_path,
            dfs_final_path,
            screenshot_mode=True
        )

        self.assertGreater(len(dfs_final_path), len(bfs_final_path))
        print(
            f"{EscapeSequenceColors.GREEN.value}DFS: Length of the path: {len(dfs_final_path)-1}{Style.reset}\n"
            f"{EscapeSequenceColors.GREEN.value}BFS: Length of the path: {len(bfs_final_path)-1}{Style.reset}\n"
        )

    def test_dfs_makes_less_steps(self):
        maze = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        start_position = (4, 4)
        finish_position = (2, 0)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        bfs_algo_steps, bfs_final_path, bfs_history, _ = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        dfs_algo_steps, dfs_final_path, dfs_history, _ = dfs_maze.search()

        visualize_data(
            bfs_history,
            dfs_history,
            maze,
            start_position,
            finish_position,
            bfs_final_path,
            dfs_final_path,
            screenshot_mode=True
        )

        self.assertGreater(bfs_algo_steps, dfs_algo_steps)
        print(
            f"{EscapeSequenceColors.GREEN.value}DFS: Number of steps needed to find the path: {dfs_algo_steps}{Style.reset}\n"
            f"{EscapeSequenceColors.GREEN.value}BFS: Number of steps needed to find the path: {bfs_algo_steps}{Style.reset}\n"
        )

    def test_bfs_makes_less_steps(self):
        maze = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        start_position = (4, 4)
        finish_position = (3, 4)
        bfs_maze = Search(maze, start_position, finish_position, "bfs")
        bfs_algo_steps, bfs_final_path, bfs_history, _ = bfs_maze.search()
        dfs_maze = Search(maze, start_position, finish_position, "dfs")
        dfs_algo_steps, dfs_final_path, dfs_history, _ = dfs_maze.search()

        visualize_data(
            bfs_history,
            dfs_history,
            maze,
            start_position,
            finish_position,
            bfs_final_path,
            dfs_final_path,
            screenshot_mode=True
        )

        self.assertGreater(dfs_algo_steps, bfs_algo_steps)
        print(
            f"{EscapeSequenceColors.GREEN.value}DFS: Number of steps needed to find the path: {dfs_algo_steps}{Style.reset}\n"
            f"{EscapeSequenceColors.GREEN.value}BFS: Number of steps needed to find the path: {bfs_algo_steps}{Style.reset}\n"
        )


if __name__ == "__main__":
    unittest.main()
