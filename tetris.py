import random
import pygame, sys
from copy import copy
from pygame.locals import *

BOARDX = 12
BOARDY = 21
BLOCKSIZE = 20
x_top_left = 30
y_top_left = 30

level = 0
lines_per_level = 0

def display_board(board, x, y, piece_array, score, shadow_y, DISPLAY):
    for i in range(0, BOARDY):
        for j in range(0, BOARDX):
            if not board[i][j]:
                if i-y >= 0 and i-y < 4 and j-x >= 0 and j-x < 4 and piece_array[i-y][j-x] == 1:
                    pygame.draw.rect(DISPLAY,(255, 255, 255),(x_top_left+j*BLOCKSIZE, y_top_left+i*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))
                elif i-shadow_y >= 0 and i-shadow_y < 4 and j-x >= 0 and j-x < 4 and piece_array[i-shadow_y][j-x] == 1:
                    pygame.draw.rect(DISPLAY,(128, 128, 128),(x_top_left+j*BLOCKSIZE, y_top_left+i*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))
                else:
                    pygame.draw.rect(DISPLAY,(0, 0, 0),(x_top_left+j*BLOCKSIZE, y_top_left+i*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))    
            elif board[i][j] == 1:
                pygame.draw.rect(DISPLAY,(255, 255, 255),(x_top_left+j*BLOCKSIZE, y_top_left+i*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))

def init_board():
    board = []
    for i in range (0, BOARDY):
        board.append([0]*BOARDX)
        board[i][0] = 2
        board[i][BOARDX-1] = 2
    
    for i in range(0, BOARDX):
        board[BOARDY-1][i] = 2
    
    return board

def rotate_piece(piece_array):
    for i in range(0, len(piece_array)):
        for j in range(0, i):
            piece_array[i][j], piece_array[j][i] = piece_array[j][i], piece_array[i][j]
    
    for i in range(0, len(piece_array)):
        for j in range(0, len(piece_array) // 2):
            piece_array[i][j], piece_array[i][len(piece_array)-1-j] =  piece_array[i][len(piece_array)-1-j], piece_array[i][j]

def reset_piece_array(piece_array):
    for i in range(0, len(piece_array)):
        for j in range(0, len(piece_array)):
            piece_array[i][j] = 0

def check_piece(piece, piece_tracker):
    for i in range(0, 4):
        if piece == piece_tracker[i]:
            return 0
    return 1

def gen_piece(piece_array, piece_tracker):
    random.seed()
    piece = random.randint(0, 6)
    while not check_piece(piece, piece_tracker):
        piece = random.randint(0, 6)
    
    for i in range(1, 4):
        piece_tracker[i-1] = piece_tracker[i]
    piece_tracker[3] = piece

    reset_piece_array(piece_array)

    if(piece == 0):
        piece_array[0][1] = 1
        piece_array[1][1] = 1
        piece_array[2][1] = 1
        piece_array[3][1] = 1
    if(piece == 1):
        piece_array[0][0] = 1
        piece_array[1][0] = 1
        piece_array[1][1] = 1
        piece_array[1][2] = 1
    if(piece == 2):
        piece_array[0][2] = 1
        piece_array[1][0] = 1
        piece_array[1][1] = 1
        piece_array[1][2] = 1
    if(piece == 3):
        piece_array[0][2] = 1
        piece_array[0][3] = 1
        piece_array[1][2] = 1
        piece_array[1][3] = 1
    if(piece == 4):
        piece_array[0][2] = 1
        piece_array[0][3] = 1
        piece_array[1][1] = 1
        piece_array[1][2] = 1
    if(piece == 5):
        piece_array[1][1] = 1
        piece_array[1][2] = 1
        piece_array[1][3] = 1
        piece_array[2][2] = 1
    if(piece == 6):
        piece_array[0][1] = 1
        piece_array[0][2] = 1
        piece_array[1][2] = 1
        piece_array[1][3] = 1

def check_collision(piece_array, x, y, board):
    check = 0
    squares = []
    for i in range(0, len(piece_array)):
        squares.append([0, 0])
    k = 0
    for i in range(0, len(piece_array)):
        for j in range(0, len(piece_array)):
            if piece_array[i][j] == 1:
                squares[k][1] = i
                squares[k][0] = j
                k += 1
    
    for i in range(0, len(piece_array)):
        if squares[i][0]+x > 0 and squares[i][0]+x < BOARDX and squares[i][1]+y < BOARDY:
            if not board[squares[i][1]+y][squares[i][0]+x] == 0:
                return 1
        else:
            return 1
    
    return 0

def add_to_board(board, x, y, piece_array):
    for i in range(y, len(piece_array)+y):
        for j in range(x, len(piece_array)+x):
            if(piece_array[i-y][j-x] == 1):
                board[i][j] = 1           

def check_lines(board, x, y):
    global lines_per_level
    global level
    lines = 0
    start_clear = -1

    for i in range(y, min(y+4, BOARDY-1)):
        check = 1
        for j in range(1, BOARDX-1):
             if board[i][j] == 0:
                 check = 0
                 break
        if check:
            if lines == 0:
                start_clear = i-1
            lines_per_level += 1
            lines += 1
            if lines_per_level == 10:
                lines_per_level = 0
                level += 1
            for j in range(1, BOARDX-1):
                board[i][j] = 0

    if lines:
        for i in range(start_clear, -1, -1):
            for j in range(0, BOARDX):
                board[i+lines][j] = board[i][j]

    #the following is used if training is done with the fitness function being the number of lines cleared instead of the score
    #return lines

    if lines == 1:
        return (level+1)*40
    elif lines == 2:
        return (level+1)*100
    elif lines == 3:
        return (level+1)*300
    elif lines == 4:
        return (level+1)* 1200
    else:
        return 0


def evaluate_move(weights, board, piece_array):
    piece = piece_array[:]
    best_rotation = 0
    best_x = 0
    best_score = float('-inf')
    best_params = [0, 0, 0, 0]
    for i in range(0, 4):
        for x in range(-1, BOARDX-1):
            params = [0, 0, 0, 0]
            sample_board = [row[:] for row in board]
            y = 0
            score = float("-inf")
            if not check_collision(piece, x, y, sample_board):
                while not check_collision(piece, x, y, sample_board):
                    y += 1
                y -= 1
                bad_move = 0
                for j in range(0, len(piece_array)):
                    for k in range(0, len(piece_array)):
                        if piece[j][k] == 1 and k+x >= BOARDX-1:
                            bad_move = 1
                    
                if not bad_move:
                    add_to_board(sample_board, x, y, piece)
                    params = calculate_params(sample_board, params)
                    score = weights[0] * params[0] + weights[1]*params[1] + weights[2]*params[2] + weights[3]*params[3]
                    if(score > best_score):
                        best_score = score
                        best_rotation = i
                        best_x = x
                        best_params = params[:]
        rotate_piece(piece)
    return (best_rotation, best_x)

def calculate_params(board, params):
    #aggregate height
    for i in range(4):
        params[i] = 0

    
    for x in range(1, BOARDX-1):
        y = 0
        while board[y][x] == 0:
            y += 1
        params[0] += BOARDY-y-1

    #number of complete lines
    for y in range(0, BOARDY-1):
        count = 0
        for x in range(1, BOARDX-1):
            if board[y][x] == 1:
                count += 1
        if count == BOARDX-2:
            params[1] += 1
    
    #number of holes
    for y in range(1, BOARDY-1):
        for x in range(1, BOARDX-1):
            if board[y][x] == 0 and board[y-1][x] != 0 and board[y+1][x] != 0:
                params[2] += 1

    #overall roughness/"bumpiness" of the board
    height = 0
    y = 0
    while board[y][1] == 0:
        y += 1
    height = BOARDY - y - 1
    
    for x in range(2, BOARDX-1):
        height_curr = 0
        y = 0
        while board[y][x] == 0:
            y += 1
        height_curr = BOARDY - y - 1
        params[3] += abs(height_curr-height)
        height = height_curr

    return params

def norm_vector(v):
    length = 0
    for i in v:
        length += i ** 2
    length = length ** 0.5
    for i in range(0, len(v)):
        v[i] /= length

def train():
    population = 60
    num_games = 20
    vectors = []
    average_score = [0]*population
    sub_group_portion = 3
    num_generations = 80
    mutation_chance = 7 #percent
    best_fitness = 0
    best_vector = [0]*4

    #initializing the genomes
    for i in range(0, population):
        vectors.append([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        norm_vector(vectors[i])
    print("STARTING TRAINING")
    for i in range(0, num_generations):
        average_score = [0]*population
        #calculating the average score for each genome
        for j in range(0, population):
            for k in range(0, num_games):
                average_score[j] += play(vectors[j])
            average_score[j] /= num_games

        children = []
        #generating children by combining the genes of parents while also allowing for mutations
        for j in range(0,  population // sub_group_portion):
            #weighted average of parents genes with respect to their performance
            sub_group = (random.sample(average_score, population // sub_group_portion))
            sub_group.sort(reverse=True)
            v1 = vectors[average_score.index(sub_group[0])]
            v2 = vectors[average_score.index(sub_group[1])]
            children.append([sub_group[0] * v1[k] + sub_group[1] * v2[k] for k in range(0, len(v1))])
            norm_vector(children[j])     

            #5% chance of mutation
            mutation = random.randrange(100)
            if(mutation  <= mutation_chance):
                mutation = random.randint(0, 3)
                mutation_type = random.uniform(-0.25, 0.25)
                children[j][mutation] += mutation_type
            norm_vector(children[j])

        #getting rid of the lowest 30%
        vectors_2 = [x for _,x in sorted(zip(average_score,vectors), reverse=True)]
        average_score.sort(reverse=True)
        vectors = [row[:] for row in vectors_2]
        del vectors[len(vectors)-len(children):]

        #adding in the children for the next generation
        for j in children:
            vectors.append(j)
        if(average_score[0] > best_fitness):
            best_fitness = average_score[0]
            best_vector = vectors[0][:]
        print("GENERATION: {}   BEST FITNESS: {}".format(i, average_score[0]))
        print("TOP 5 GENOMES")
        for j in range(5):
            print(vectors[j])
        #print(average_score)

    return vectors[0]
        

def play(weights):    
    level = 0
    lines_per_level = 0

    play_on = 1
    piece_generated = 0
    num_piece_generated = 0
    piece = 0
    piece_tracker = [0, 0, 0, 0]
    piece_array = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    x = 0
    y = 0
    score = 1
    board = init_board()

    random.seed()

    moves = (0, 0)

    while play_on and num_piece_generated < 500:
        if not piece_generated:
            gen_piece(piece_array, piece_tracker)
            piece_generated = 1
            num_piece_generated += 1
            x = 3
            y = 0
            if check_collision(piece_array, x, y, board):
                play_on = 0
            else:
                moves = evaluate_move(weights, board, piece_array)
                for i in range(0, moves[0]):
                   rotate_piece(piece_array)
        else:
            if(moves[1] != x and not check_collision(piece_array, moves[1], y, board)):
                x = moves[1]
            while not check_collision(piece_array, x, y+1, board):
                    y += 1

            add_to_board(board, x, y, piece_array)
            #this function is called during training, and so we set the score to the number of lines cleared
            score += check_lines(board, x, y)
            piece_generated = 0

    return score 

#debugging purposes
def print_board(board):
    for i in range(0, BOARDY):
        for j in range(0, BOARDX):
            if(board[i][j] == 0):
                print(" ", end="", flush=True)
            elif(board[i][j] == 1):
                print("X", end="", flush=True)
            else:
                print("#", end="", flush=True)
        print()

def play_animate(weights):
    pygame.init()

    DISPLAY=pygame.display.set_mode((400,500),0,32)

    WHITE=(255,255,255)

    DISPLAY.fill((0, 0, 0))

    pygame.draw.rect(DISPLAY,WHITE,(x_top_left, y_top_left, BLOCKSIZE, BOARDY*BLOCKSIZE))
    pygame.draw.rect(DISPLAY,WHITE,(x_top_left+BLOCKSIZE, y_top_left+(BOARDY-1)*BLOCKSIZE, (BOARDX-1)*BLOCKSIZE, BLOCKSIZE))
    pygame.draw.rect(DISPLAY,WHITE,(x_top_left+(BOARDX-1)*BLOCKSIZE, y_top_left, BLOCKSIZE, BOARDY*BLOCKSIZE))
    pygame.display.update()
    
    level = 0
    lines_per_level = 0

    play_on = 1
    piece_generated = 0
    piece = 0
    piece_tracker = [0, 0, 0, 0]
    piece_array = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    x = 1
    y = 0
    shadow_y = 0
    frame = 0
    frame_cycle = 1
    score = 0
    board = init_board()

    random.seed()
    clock = pygame.time.Clock()

    moves = (0, 0)

    while play_on:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if not piece_generated:
            gen_piece(piece_array, piece_tracker)
            piece_generated = 1
            x = 3
            y = 0
            if check_collision(piece_array, x, y, board):
                play_on = 0
            else:
                moves = evaluate_move(weights, board, piece_array)

                for i in range(0, moves[0]):
                   rotate_piece(piece_array)

                shadow_y = 0
                while not check_collision(piece_array, x, shadow_y, board):
                    shadow_y += 1
        else:
            if(moves[1] != x and not check_collision(piece_array, moves[1], y, board)):
                x = moves[1]
                #the following two lines of code will animate the pieces moving left to right 
                #has the downside of reducing performance in certain cases where pieces are already stacked high
                #x += moves[1] > x
                #x -= moves[1] < x
            if not check_collision(piece_array, x, y+1, board):
                frame = (frame + 1) % frame_cycle
                if frame == 0:
                    y += 1
            else:
                add_to_board(board, x, y, piece_array)
                score += check_lines(board, x, y)
                pygame.display.set_caption('SCORE: '+str(score))
                piece_generated = 0
                frame = 0

            shadow_y = y
            while not check_collision(piece_array, x, shadow_y, board):
                shadow_y += 1 
        if frame == 0:
            display_board(board, x, y, piece_array, score, shadow_y-1, DISPLAY)
            pygame.display.update()
        clock.tick(120)
    print(score)

if __name__ == "__main__":
    play_animate([-0.52, 0.55, -0.64, -0.15])
