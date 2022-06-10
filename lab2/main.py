from mpi4py import MPI
import random
from time import sleep
from os import getpid
from board import Board
import time
from copy import deepcopy
import random

CPU   = 1
HUMAN = 2

DEPTH = 7

DATA_TAG = 1
TASK_TAG = 2
END_TAG  = 3
NONE_TAG = 4
# FINAL_TAG = 5

def evaluate(board, depth=DEPTH):
    if (board.game_end()): 
        return 1 if board.last_mover == CPU else -1
    
    if depth == 0: return 0

    new_mover = HUMAN if board.last_mover == CPU else CPU
    
    res = 0
    total = 0
    moves = 0
    all_win, all_lose = True, True
    
    for col_i in range(board.cols):
        if board.move_legal(col_i):
            moves += 1

            board.move(col_i, new_mover)
            res = evaluate(board, depth - 1)
            board.undo_move(col_i)

            if res  > -1: all_lose = False
            if res !=  1: all_win  = False
            if res ==  1 and new_mover == CPU:   return  1
            if res == -1 and new_mover == HUMAN: return -1

            total += res

    if all_win:  return  1
    if all_lose: return -1

    return total/board.cols



def fill_q_and_assign(board):
    # generiranje zadataka
    q = []
    for i in range(board.cols):
        for j in range(board.cols):
            q.append((i, j))

    totals = {i: 0 for i in range(board.cols)}
    mask = [0 for _ in range(board.cols)]
    
    start = time.time()

    all_solved = False
    count_solved = board.cols**2
    while not all_solved:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, status=status)
        
        tag = status.Get_tag()
        if tag == TASK_TAG:
            # dohvati posiljatelja, izvuci zadatak i posalji mu ga
            sender = status.Get_source()

            if q != []:
                task = q.pop()
                comm.send(task, dest=sender, tag=TASK_TAG)
            else: 
                comm.send(None, dest=sender, tag=NONE_TAG)

        elif tag == DATA_TAG:
            # posalji kopiju ploce
            sender = status.Get_source()
            comm.send(deepcopy(board), dest=sender, tag=DATA_TAG)

        elif tag == END_TAG:
            # dohvati rezultat posla i kojem zadatku s prve razine pripada
            res, task_i, finished_in_two = msg
            if finished_in_two:
                mask[task_i] = res
                totals[task_i] = res

            elif mask[task_i] == 0:
                totals[task_i] += res / board.cols
                
            count_solved -= 1

        # ako su svi rjeseni
        if count_solved == 0:
            all_solved = True

    end = time.time()
    print(f'TIME: {end - start}s')

    argmax_cols = [i for i, elem in enumerate(totals) if elem == max(totals, key=totals.get) and board.height[i] < board.rows]
    return argmax_cols[0]


def receive_task_and_calculate():
    # dohvati plocu
    comm.send(None, dest=0, tag=DATA_TAG)

    board_copy = comm.recv(source=0, tag=DATA_TAG)
    
    # salji zahtjev za zadatak
    comm.send(None, dest=0, tag=TASK_TAG)

    # primi zadatak, napravi poteze
    status = MPI.Status()
    task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
    tag = status.Get_tag()
    if tag == NONE_TAG:
        return
    
    # napravi prvi potez, provjeri je li pobjeda
    board_copy.move(task[0], CPU)
    if board_copy.game_end():
        comm.send((1, task[0], True), dest=0, tag=END_TAG)
        return 

    # napravi drugi potez, provjeri je li pobjeda
    board_copy.move(task[1], HUMAN)
    if board_copy.game_end():
        comm.send((-1, task[0], True), dest=0, tag=END_TAG)
        return 

    # depth - 2 zbog generiranja 49 zadataka
    res = evaluate(board_copy, depth=DEPTH-2)

    # vrati rezultat
    comm.send((res, task[0], False), dest=0, tag=END_TAG)


comm = MPI.COMM_WORLD
index = comm.Get_rank()
size = comm.Get_size()


def main():
    # glavni proces
    if index == 0 or size == 1:
        board = Board()
        print('Pokretanje . . .')
        print(board)

        while True:
            # dohvati igracev potez i odigraj na ploci
            input_col = int(input('Upisite potez:\n'))
            board.move(input_col, HUMAN)
            print(board)

            # provjera je li igrac pobijedio
            if board.game_end():
                print('Pobijedili ste')
                break

            # dohvati CPU potez i odigraj na ploci
            if size != 1:
                best_col = fill_q_and_assign(board)
            else:
                start = time.time()

                totals = {i:0 for i in range(board.cols)}
                for i in range(board.cols):
                    board_copy = deepcopy(board)
                    board_copy.move(i, CPU)
                    totals[i] = evaluate(board_copy, depth=DEPTH-1)

                argmax_cols = [i for i, elem in enumerate(totals) if elem == max(totals, key=totals.get) and board.height[i] < board.rows]
                best_col = argmax_cols[0]
                
                end = time.time()
                print(f'TIME: {end - start}s')
    
            print(f'CPUov potez je {best_col}')
            board.move(best_col, CPU)
            print(board)

            # provjera je li CPU pobijedio
            if board.game_end():
                print('Izgubili ste')
                break

    # ostali procesi
    else:
        while True:
            receive_task_and_calculate()

main()