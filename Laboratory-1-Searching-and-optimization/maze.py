from enum import Enum
import numpy

class NodeState(Enum):
    NOT_VISITED = 0
    WALL = 1
    # START = 2
    # FINISH = 3
    VISITED = 4

class Node:
    def __init__(self, state: tuple[int, int], parent):
        self.state = state
        self.parent = parent


class StackFrontier:
    def __init__(self):
        self.frontier: list[Node] = []

    def add(self, node: Node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
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
    ):

        if search_type not in ["bfs", "dfs"]:
            raise ValueError("Invalid search type")

        self.search_type = search_type

        self.maze = maze

        self.height = len(maze)

        if self.height == 0:
            raise ValueError("Maze is empty")

        self.width = len(maze[0])

        if self.width == 0:
            raise ValueError("Maze is empty")

        self.start = start

        if (
            start[0] < 0
            or start[0] >= self.height
            or start[1] < 0
            or start[1] >= self.width
        ):
            raise ValueError("Start position is out of bounds")

        if maze[start[0]][start[1]] == NodeState.WALL.value:
            raise ValueError("Start position is a wall")

        self.finish = finish

        if (
            start[0] < 0
            or start[0] >= self.height
            or start[1] < 0
            or start[1] >= self.width
        ):
            raise ValueError("Finish position is out of bounds")

        if maze[start[0]][start[1]] == NodeState.WALL.value:
            raise ValueError("Finish position is a wall")

        self.step_number = 0

        self.performed_steps_history: list[tuple[int, tuple[int, int]]] = []

        self.history: tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]] = (
            (self.height, self.width),
            self.performed_steps_history,
        )

    def find_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Find current tile neighbors, only horizontal and vertical, diagonal tiles are forbidden

        Returns:
            _type_: _description_
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
        self.maze[cell[0]][cell[1]] = NodeState.VISITED.value

    # def mark_start(self, cell: tuple[int, int]):
    #     self.maze[cell[0]][cell[1]] = NodeState.START.value

    # def mark_finish(self, cell: tuple[int, int]):
    #     self.maze[cell[0]][cell[1]] = NodeState.FINISH.value

    def search(
        self,
    ) -> tuple[int, tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]], Node]:

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
                return -1, self.history, last_node

            current_node = frontier.remove()
            last_node = current_node
            self.mark_visited(current_node.state)
            self.performed_steps_history.append((self.step_number, current_node.state))

            if current_node.state == self.finish:
                return self.step_number, self.history, last_node

            for neighbor in self.find_neighbors(current_node.state):
                if neighbor not in explored:
                    new_node = Node(state=neighbor, parent=current_node)
                    frontier.add(new_node)
                    explored.add(neighbor)

            self.step_number += 1

    def get_final_path(self, node: Node) -> list[tuple[int, int]]:
        path = []
        if self.step_number != -1:
            while node.parent is not None:
                path.append(node.state)
                node = node.parent
            path.append(self.start)
            path.reverse()
        return path

def generate_maze(size=(3,3), empty_per_wall = 5):

    generated = numpy.random.randint(empty_per_wall, size=size)
    generated += 1
    generated[generated > 1] = 0
    return generated.tolist()

def random_coordinate(size):
    return tuple(numpy.random.randint(0, size, 2))