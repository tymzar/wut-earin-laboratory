from enum import Enum
import numpy
from copy import deepcopy


class NodeState(Enum):
    NOT_VISITED = 0
    WALL = 1
    VISITED = 2


class Node:
    def __init__(self, state: tuple[int, int], parent):
        """Node constructor to initialize the state and parent of the node

        Args:
            state (tuple[int, int]): State of the node in format (row, column)
            parent (Node): Parent of the node
        """

        self.state = state
        self.parent = parent


class StackFrontier:
    def __init__(self) -> None:
        """StackFrontier constructor to initialize DFS frontier"""

        self.frontier: list[Node] = []

    def add(self, node: Node) -> None:
        """StackFrontier add method to add a node to the frontier

        Args:
            node (Node): Node to be added to the frontier
        """
        self.frontier.append(node)

    def contains_state(self, state: tuple[int, int]) -> bool | None:
        """StackFrontier contains_state method to check if the state is in the frontier

        Args:
            state (tuple[int, int]): State to be checked

        Returns:
            UnionType[bool, None]: True if the state is in the frontier, False otherwise
        """
        return any(node.state == state for node in self.frontier)

    def empty(self) -> bool:
        """StackFrontier empty method to check if the frontier is empty

        Returns:
            bool: True if the frontier is empty, False otherwise
        """

        return len(self.frontier) == 0

    def remove(self) -> Node:
        """StackFrontier remove method to remove the last element of the frontier

        Raises:
            Exception: If the frontier is empty

        Returns:
            Node: Lastly removed node
        """
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        """QueueFrontier remove method to remove the first element of the frontier

        Raises:
            Exception: If the frontier is empty

        Returns:
            _type_: Lastly removed node
        """
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node


class Search:

    def __init__(
        self,
        maze: list[list[int]],
        start: tuple[int, int],
        finish: tuple[int, int],
        search_type: str = "bfs",
    ) -> None:
        """Constructor for the Search class that will perform a search on the given maze

        Args:
            maze (list[list[int]]): Maze to be searched containing 0 for empty cells and 1 for walls
            start (tuple[int, int]): Start position in the maze in format (row, column)
            finish (tuple[int, int]): Finish position in the maze in format (row, column)
            search_type (str, optional): Type of search to be performed, either "bfs" or "dfs". Defaults to "bfs".

        Raises:
            ValueError: If the search type is not "bfs" or "dfs"
            ValueError: If the maze is empty
            ValueError: If the start position is out of bounds
            ValueError: If the finish position is out of bounds
            ValueError: If the start position is a wall
            ValueError: If the finish position is a wall
            ValueError: If the start position is the same as the finish position
            ValueError: If the search type is not "bfs" or "dfs"
        """

        if search_type not in ["bfs", "dfs"]:
            raise ValueError("Invalid search type")

        self.search_type = search_type

        self.maze = deepcopy(maze)

        self.height = len(self.maze)

        if self.height == 0:
            raise ValueError("Maze is empty")

        if any(len(row) != len(self.maze[0]) for row in self.maze):
            raise ValueError("Maze is not rectangular")

        if any(cell not in [0, 1] for row in self.maze for cell in row):
            raise ValueError("Maze contains invalid cell value")

        if len(self.maze[0]) == 0:
            raise ValueError("Maze is empty")

        self.width = len(self.maze[0])

        if self.width == 0:
            raise ValueError("Maze is empty")

        if (
            start[0] < 0
            or start[0] >= self.height
            or start[1] < 0
            or start[1] >= self.width
        ):
            raise ValueError("Start position is out of bounds")

        if self.maze[start[0]][start[1]] == NodeState.WALL.value:
            raise ValueError("Start position is a wall")

        self.start = start

        if (
            finish[0] < 0
            or finish[0] >= self.height
            or finish[1] < 0
            or finish[1] >= self.width
        ):
            raise ValueError("Finish position is out of bounds")

        if self.maze[finish[0]][finish[1]] == NodeState.WALL.value:
            raise ValueError("Finish position is a wall")

        self.finish = finish

        if start == finish:
            raise ValueError("Start position is the same as the finish position")

        self.step_number = 0

        self.performed_steps_history: list[tuple[int, tuple[int, int]]] = []

        self.history: tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]] = (
            (self.height, self.width),
            self.performed_steps_history,
        )

    def find_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Find the vertical and horizontal neighbors of the cell

        Args:
            cell (tuple[int, int]): Cell to find the neighbors

        Returns:
            list[tuple[int, int]]: List of tuples with the neighbors of the cell
        """

        neighbors = []

        for i in range(1, -2, -1):
            for j in range(1, -2, -1):
                if (i == 0 and j == 0) or (i != 0 and j != 0):
                    continue

                current_row = cell[0] - i
                current_cell = cell[1] - j

                if (current_row >= 0 and current_row <= self.height - 1) and (
                    current_cell >= 0 and current_cell <= self.width - 1
                ):
                    if self.maze[current_row][current_cell] != 1:
                        neighbors.append((current_row, current_cell))

        return neighbors

    def mark_visited(self, cell: tuple[int, int]):
        """Mark the cell as visited

        Args:
            cell (tuple[int, int]): Cell to be marked as visited
        """

        self.maze[cell[0]][cell[1]] = NodeState.VISITED.value

    def search(
        self,
    ) -> tuple[
        int,
        list[tuple[int, int]],
        tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]],
        Node,
    ]:
        """Search for the finish position in the maze using either BFS or DFS.

        Returns:
            tuple[ int, list[tuple[int, int]], tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]], Node, ]: Amount of steps, final path, history, and finish node
        """

        start_node = Node(state=self.start, parent=None)
        last_node = start_node

        frontier = QueueFrontier()

        if self.search_type == "dfs":
            frontier = StackFrontier()

        frontier.add(start_node)

        explored = set()

        while True:
            if frontier.empty():
                self.step_number = -1
                break

            current_node = frontier.remove()
            last_node = current_node
            self.mark_visited(current_node.state)
            self.performed_steps_history.append((self.step_number, current_node.state))

            if current_node.state == self.finish:
                break

            for neighbor in self.find_neighbors(current_node.state):
                if neighbor not in explored:
                    new_node = Node(state=neighbor, parent=current_node)
                    frontier.add(new_node)
                    explored.add(neighbor)

            self.step_number += 1

        final_path = self.get_final_path(last_node)
        steps_amount = len(final_path) if self.step_number != -1 else -1
        return steps_amount, final_path, self.history, last_node

    def get_final_path(self, node: Node) -> list[tuple[int, int]]:
        """Get the final path from the start to the finish node

        Args:
            node (Node): Finish node

        Returns:
            list[tuple[int, int]]: List of tuples with the path from the start to the finish node
        """

        path = []
        if self.step_number != -1:
            while node.parent is not None:
                path.append(node.state)
                node = node.parent
            path.append(self.start)
            path.reverse()
        return path


def generate_maze(
    size=(3, 3), empty_per_wall=5
) -> tuple[list[list[int]], tuple[int, int], tuple[int, int]]:
    """Generate a random maze with a start and finish position of the given size and empty cells per wall ratio

    Args:
        size (tuple, optional): Grid size in format (rows, columns). Defaults to (3, 3).
        empty_per_wall (int, optional): Amount of empty cells per wall. Defaults to 5.

    Returns:
        tuple[list[list[int]], tuple[int, int], tuple[int, int]]: Maze, start position, and finish position
    """

    temporary_maze = numpy.zeros(size)
    cells_amount = size[0] * size[1]

    while True:
        start = random_coordinate(size)
        finish = random_coordinate(size)
        if start != finish:
            break

    while True:
        wall_candidate = random_coordinate(size)
        if wall_candidate != start and wall_candidate != finish:
            temporary_maze[wall_candidate] = NodeState.WALL.value
            if numpy.count_nonzero(temporary_maze == NodeState.WALL.value) == int(
                cells_amount / empty_per_wall
            ):
                break

    return temporary_maze, start, finish


def random_coordinate(size: tuple[int, int]) -> tuple[int, int]:
    """Generate a random coordinate within the maze size

    Args:
        size (tuple[int, int]): Grid size in format (rows, columns)

    Returns:
        tuple[int, int]: Random coordinate within the grid size
    """
    return tuple(numpy.random.randint(0, size, 2))
