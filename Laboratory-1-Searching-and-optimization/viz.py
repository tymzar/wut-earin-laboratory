import pygame
import time
from maze import NodeState

def visualize_data(
    history: tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]],
    grid_size: tuple[int, int],
    maze: list[list[int]],
    start_position: tuple[int, int],
    finish_position: tuple[int, int],
    final_path: list[tuple[int, int]],
) -> None:

    HEIGHT, WIDTH = grid_size

    # Colors
    BLACK = (0, 0, 0)
    GRAY = (180, 180, 180)
    WHITE = (255, 255, 255)

    # Create game
    pygame.init()
    size = width, height = HEIGHT * 100, WIDTH * 100
    screen = pygame.display.set_mode(size)

    # Fonts
    OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
    smallFont = pygame.font.Font(OPEN_SANS, 20)
    mediumFont = pygame.font.Font(OPEN_SANS, 28)
    largeFont = pygame.font.Font(OPEN_SANS, 40)

    # Compute board size
    BOARD_PADDING = 20
    board_width = ((2 / 3) * width) - (BOARD_PADDING * 2)
    board_height = height - (BOARD_PADDING * 2)
    cell_size = int(min(board_width / WIDTH, board_height / HEIGHT))
    board_origin = (BOARD_PADDING, BOARD_PADDING)

    wall = pygame.image.load("assets/images/wall.png")
    wall = pygame.transform.scale(wall, (cell_size, cell_size))

    flag = pygame.image.load("assets/images/flag.png")
    flag = pygame.transform.scale(flag, (cell_size, cell_size))

    start = pygame.image.load("assets/images/start.png")
    start = pygame.transform.scale(start, (cell_size, cell_size))

    print("Visualize data...")

    size, step_history = history

    while True:

        cells = []
        for i in range(HEIGHT):
            row = []
            for j in range(WIDTH):

                # Draw rectangle for cell
                rect = pygame.Rect(
                    board_origin[0] + j * cell_size,
                    board_origin[1] + i * cell_size,
                    cell_size,
                    cell_size,
                )
                pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, WHITE, rect, 3)

                if maze[i][j] == NodeState.WALL.value:
                    screen.blit(wall, rect)

                if (i, j) == finish_position:
                    screen.blit(flag, rect)

                if (i, j) == start_position:
                    screen.blit(start, rect)

                row.append(rect)
            cells.append(row)

        for search_step in step_history:

            time.sleep(0.5)

            _, cell = search_step

            if cell == finish_position or cell == start_position:
                continue

            rect = cells[cell[0]][cell[1]]

            pygame.draw.rect(screen, (0, 0, 255), rect)

            pygame.display.flip()
            time.sleep(0.5)

            # print a blue dot for the current position if it is not the start or the end

        for step in final_path:
            rect = cells[step[0]][step[1]]
            pygame.draw.rect(screen, (0, 255, 0), rect)
            pygame.display.flip()
            time.sleep(0.5)

        pygame.display.flip()
