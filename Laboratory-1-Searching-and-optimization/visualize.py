import pygame
import time
from maze import NodeState


def visualize_data(
    first_history: tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]],
    second_history: tuple[tuple[int, int], list[tuple[int, tuple[int, int]]]],
    grid_size: tuple[int, int],
    maze: list[list[int]],
    start_position: tuple[int, int],
    finish_position: tuple[int, int],
    first_final_path: list[tuple[int, int]],
    second_final_path: list[tuple[int, int]]
) -> None:

    HEIGHT, WIDTH = grid_size

    # Colors
    BLACK = (0, 0, 0)
    GRAY = (180, 180, 180)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # Create game
    pygame.init()
    size = width, height = HEIGHT * 100, WIDTH * 100
    screen = pygame.display.set_mode(size)
    screen.fill(GRAY)


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

    _, first_steps = first_history
    _, second_steps = second_history

    def set_items(i, j, rect, surface):
        if maze[i][j] == NodeState.WALL.value:
            surface.blit(wall, rect)

        if (i, j) == finish_position:
            surface.blit(flag, rect)

        if (i, j) == start_position:
            surface.blit(start, rect)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                pygame.quit()
                return

        cells = []
        first_surface = pygame.Surface(screen.get_size(),  pygame.SRCALPHA)
        first_surface.set_alpha(50)
        second_surface = pygame.Surface(screen.get_size(),  pygame.SRCALPHA)
        second_surface.set_alpha(50)
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

                set_items(i, j, rect, screen)

                row.append(rect)
            cells.append(row)

        num_steps = max(len(first_steps), len(second_steps))

        for step_num in range(0, num_steps):
            
            _, first_cell = first_steps[min(len(first_steps)-1, step_num)]
            _, second_cell = second_steps[min(len(second_steps)-1, step_num)]


            first_rect = cells[first_cell[0]][first_cell[1]]
            second_rect = cells[second_cell[0]][second_cell[1]]

            pygame.draw.rect(first_surface, BLUE, first_rect)
            pygame.draw.rect(second_surface, RED, second_rect)
            set_items(first_cell[0], first_cell[1], first_rect, first_surface)
            set_items(second_cell[0], second_cell[1], second_rect, second_surface)
            screen.blit(second_surface, (0, 0))
            screen.blit(first_surface, (0, 0))
            pygame.display.flip()
            time.sleep(0.1)

            # print a blue dot for the current position if it is not the start or the end


        num_final_steps = max(len(first_final_path), len(second_final_path))
        for step_num in range(0, num_final_steps):
            first_step = first_final_path[min(len(first_final_path) - 1, step_num)]
            second_step = second_final_path[min(len(second_final_path) - 1, step_num)]
            first_rect = cells[first_step[0]][first_step[1]]
            second_rect = cells[second_step[0]][second_step[1]]
            pygame.draw.rect(first_surface, GREEN, first_rect)
            set_items(first_step[0], first_step[1], first_rect, first_surface)
            pygame.draw.rect(second_surface, BLACK, second_rect)
            set_items(second_step[0], second_step[1], second_rect, second_surface)
            screen.blit(second_surface, (0, 0))
            screen.blit(first_surface, (0, 0))
            pygame.display.flip()
            time.sleep(0.2)

        pygame.display.flip()
        time.sleep(1)
