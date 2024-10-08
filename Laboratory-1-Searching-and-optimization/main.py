from maze import Search
from visualize import visualize_data
from maze import generate_maze, random_coordinate
from argparse import ArgumentParser
from colored import Fore, Style
from enum import Enum


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
        "-s",
        "--screenshot-only",
        action="store_true",
        help=f"Boolean value if to make only screenshot instead of showing the visualization e.g. --screenshot-only",
    )

    input_arguments = command_line_parser.parse_args()

    if input_arguments.maze_file and input_arguments.size:
        print(
            f"{EscapeSequenceColors.ORANGE.value}You can only use one of the options, either --size or --maze-file, in this case --maze-file will be used{Style.reset}"
        )

    if input_arguments.maze_file:
        with open(input_arguments.maze_file, "r") as file:
            start_position = tuple(map(int, file.readline().split()))
            finish_position = tuple(map(int, file.readline().split()))
            maze = [list(map(int, line.split())) for line in file.readlines()]

    elif input_arguments.size:
        maze, start_position, finish_position = generate_maze(input_arguments.size)
    else:
        raise ValueError(
            "You must use either --size or --maze-file run the program. Use --help for more information."
        )

    print(
        f"{EscapeSequenceColors.LIGHT_BLUE.value}Performing search on the following maze:{Style.reset}"
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

    bfs_steps_amount, bfs_final_path, bfs_history, _ = bfs_maze.search()
    dfs_steps_amount, dfs_final_path, dfs_history, _ = dfs_maze.search()

    print(
        f"{EscapeSequenceColors.GREEN.value}BFS: Number of steps needed to find the path: {bfs_steps_amount}{Style.reset}\n"
        f"{EscapeSequenceColors.GREEN.value}BFS: Length of the path: {len(bfs_final_path)-1}{Style.reset}"
    )
    print(
        f"{EscapeSequenceColors.GREEN.value}DFS: Number of steps needed to find the path: {dfs_steps_amount}{Style.reset}\n"
        f"{EscapeSequenceColors.GREEN.value}DFS: Length of the path: {len(dfs_final_path)-1}{Style.reset}"
    )

    visualize_data(
        bfs_history,
        dfs_history,
        maze,
        start_position,
        finish_position,
        bfs_final_path,
        dfs_final_path,
        input_arguments.screenshot_only
    )


if __name__ == "__main__":
    main()
