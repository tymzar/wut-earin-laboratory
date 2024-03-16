import pygame
import sys
import time

import tictactoe
from tictactoe import CellValues

pygame.init()
size = width, height = 600, 400

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)


screen = pygame.display.set_mode(size)

mediumFont = pygame.font.Font("OpenSans-Regular.ttf", 28)
largeFont = pygame.font.Font("OpenSans-Regular.ttf", 40)
moveFont = pygame.font.Font("OpenSans-Regular.ttf", 60)


user = None
board = tictactoe.get_initial_state()
ai_turn = False


while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(GRAY)

    if user is None:
        title = largeFont.render("Play Tic-Tac-Toe", True, BLACK)
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 50)
        screen.blit(title, titleRect)

        playXButton = pygame.Rect(
            (width / 2) - width / 8, (height / 4) + 120, width / 4, 50
        )
        playX = mediumFont.render("Play as X", True, BLACK)
        playXRect = playX.get_rect()
        playXRect.center = playXButton.center
        pygame.draw.rect(screen, WHITE, playXButton)
        screen.blit(playX, playXRect)

        playOButton = pygame.Rect(
            (width / 2) - width / 8, (height / 4) + 50, width / 4, 50
        )
        playO = mediumFont.render("Play as O", True, BLACK)
        playORect = playO.get_rect()
        playORect.center = playOButton.center
        pygame.draw.rect(screen, WHITE, playOButton)
        screen.blit(playO, playORect)

        click, _, _ = pygame.mouse.get_pressed()
        if click == 1:
            mouse = pygame.mouse.get_pos()
            if playXButton.collidepoint(mouse):
                time.sleep(0.2)
                user = CellValues.X.value
            elif playOButton.collidepoint(mouse):
                time.sleep(0.2)
                user = CellValues.O.value

    else:
        tile_size = 80
        tile_origin = (width / 2 - (1.5 * tile_size), height / 2 - (1.5 * tile_size))
        tiles = []
        for i in range(3):
            row = []
            for j in range(3):
                rect = pygame.Rect(
                    tile_origin[0] + j * tile_size,
                    tile_origin[1] + i * tile_size,
                    tile_size,
                    tile_size,
                )
                pygame.draw.rect(screen, BLACK, rect, 3)

                if board[i][j] != CellValues.EMPTY.value:
                    move = moveFont.render(board[i][j], True, BLACK)
                    moveRect = move.get_rect()
                    moveRect.center = rect.center
                    screen.blit(move, moveRect)
                row.append(rect)
            tiles.append(row)

        game_over = tictactoe.is_game_finished(board)
        player = tictactoe.get_current_player(board)

        if game_over:
            winner = tictactoe.get_winner(board)
            if winner is None:
                title = f"Game Over: Tie."
            else:
                title = f"Game Over: {winner} wins."
        elif user == player:
            title = f"Play as {user}"
        else:
            title = f"Making AI move..."
        title = largeFont.render(title, True, BLACK)
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 30)
        screen.blit(title, titleRect)

        # Check for AI move
        if user != player and not game_over:
            if ai_turn:
                time.sleep(0.5)
                move = tictactoe.minimax(board)
                board = tictactoe.move(board, move)
                ai_turn = False
            else:
                ai_turn = True

        click, _, _ = pygame.mouse.get_pressed()
        if click == 1 and user == player and not game_over:
            mouse = pygame.mouse.get_pos()
            for i in range(3):
                for j in range(3):
                    if board[i][j] == CellValues.EMPTY.value and tiles[i][
                        j
                    ].collidepoint(mouse):
                        board = tictactoe.move(board, (i, j))

        if game_over:
            againButton = pygame.Rect(width / 3, height - 65, width / 3, 50)
            again = mediumFont.render("Start again", True, BLACK)
            againRect = again.get_rect()
            againRect.center = againButton.center
            pygame.draw.rect(screen, WHITE, againButton)
            screen.blit(again, againRect)
            click, _, _ = pygame.mouse.get_pressed()
            if click == 1:
                mouse = pygame.mouse.get_pos()
                if againButton.collidepoint(mouse):
                    time.sleep(0.2)
                    user = None
                    board = tictactoe.get_initial_state()
                    ai_turn = False

    pygame.display.flip()
