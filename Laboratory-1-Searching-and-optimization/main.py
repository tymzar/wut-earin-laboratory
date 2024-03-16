from maze import Search
from visualize import visualize_data
from maze import generate_maze, random_coordinate
from argparse import ArgumentParser
from colored import Fore, Style
from enum import Enum
from numpy import shape


class EscapeSequenceColors(Enum):
    ORANGE = Fore.rgb("100%", "50%", "0%")
    LIGHT_BLUE = Fore.rgb("0%", "100%", "100%")
    GREEN = Fore.rgb("0%", "100%", "0%")


def main():

    command_line_parser = ArgumentParser(
        description="Generate a maze and visualize the BFS and DFS search"
    )
    command_line_parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        help="Insert the size of the maze, e.g. --size 5 5",
    )
    command_line_parser.add_argument(
        "--maze-file",
        type=str,
        help=f"Path to the maze file e.g. --maze-file maze.txt. File format should consist of space separated integers and format should be the following: first line is the start position, second line is the finish position, and the rest of the lines are the maze",
    )
    command_line_parser.add_argument(
        "--search-type",
        type=str,
        help="Type of search to perform, either bfs or dfs",
        choices=["bfs", "dfs"],
        default="bfs",
    )

    input_arguments = command_line_parser.parse_args()

    search_type = input_arguments.search_type

    if input_arguments.maze_file and input_arguments.size:
        print(
            f"{EscapeSequenceColors.ORANGE.value}You can only use one of the options, either --size or --maze-file, in this case --maze-file will be used{Style.reset}"
        )

    if input_arguments.maze_file:
        with open(input_arguments.maze_file, "r") as file:
            start_position = tuple(map(int, file.readline().split()))
            finish_position = tuple(map(int, file.readline().split()))
            maze = [list(map(int, line.split())) for line in file.readlines()]

            size = shape(maze)

    elif input_arguments.size:
        size = input_arguments.size
        maze, start_position, finish_position = generate_maze(size)
    else:
        raise ValueError(
            "You must use either --size or --maze-file run the program. Use --help for more information."
        )

    print(
        f"{EscapeSequenceColors.LIGHT_BLUE.value}Performing serach on the following maze:{Style.reset}"
    )
    for row in maze:
        print(row)

    print(
        f"{EscapeSequenceColors.LIGHT_BLUE.value}Starting position: {start_position}{Style.reset}"
    )
    print(
        f"{EscapeSequenceColors.LIGHT_BLUE.value}Finishing position: {finish_position}{Style.reset}"
    )

    bfs_maze = Search(maze, start_position, finish_position, "bfs")
    dfs_maze = Search(maze, start_position, finish_position, "dfs")

    bfs_final_path, bfs_history, _ = bfs_maze.search()
    dfs_final_path, dfs_history, _ = dfs_maze.search()

    bfs_steps_amount = len(bfs_final_path) - 1
    dfs_steps_amount = len(dfs_final_path) - 1

    print(
        f"{EscapeSequenceColors.GREEN.value}BFS: Number of steps in path: {bfs_steps_amount}{Style.reset}"
    )
    print(
        f"{EscapeSequenceColors.GREEN.value}DFS: Number of steps in path: {dfs_steps_amount}{Style.reset}"
    )

    visualize_data(
        bfs_history,
        dfs_history,
        size,
        maze,
        start_position,
        finish_position,
        bfs_final_path,
        dfs_final_path,
    )


if __name__ == "__main__":
    main()
