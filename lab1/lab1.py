from mpi4py import MPI
import random
from time import sleep
from os import getpid

INDENT_SIZE = 4

FORK_SEND = 11
FORK_SEND_L = 12
FORK_SEND_R = 13

FORK_REQ = 14
FORK_REQ_L = 15
FORK_REQ_R = 16

OBJ = None

comm = MPI.COMM_WORLD

neighbors, forks = None, None
send = {'L': False, 'R':False}

def set_neighbors(index, size):
    global neighbors
    if index == 0:
        neighbors = {'L': size-1, 'R': index + 1}
    elif index > 0 and index < size - 1:
        neighbors = {'L': index-1, 'R': index + 1}
    elif index == size-1:
        neighbors = {'L': index - 1, 'R': 0}

def set_forks(index, size):
    global forks
    if index == 0:
        forks = {'L': 'D', 'R': 'D'}
    elif index > 0 and index < size - 1:
        forks = {'L': 'D', 'R': None}
    elif index == size-1:
        forks = {'L': None, 'R': None}
        

def check_requests(index, size):
    if comm.iprobe(source=neighbors['L'], tag=0 if size != 2 else FORK_REQ_L):
        check_neighbor_request('L', index, size)

    if comm.iprobe(source=neighbors['R'], tag=0 if size != 2 else FORK_REQ_R):
        check_neighbor_request('R', index, size)

def check_neighbor_request(side, index, size):
    msg = comm.recv(source=neighbors[side], tag=0 if size != 2 else (FORK_REQ_L if side == 'L' else FORK_REQ_R))

    if forks[side] == 'D':
        forks[side] = None
        comm.send(True, dest=neighbors[side], tag=0 if size != 2 else (FORK_SEND_L if side == 'L' else FORK_SEND_R))
    elif forks[side] == 'C':
        send[side] = True

def check_forks(side, index, size):
    other = 'R' if side == 'L' else 'L'

    # ako nemas vilicu
    if forks[side] is None:
        comm.send(True, dest=neighbors[side], tag=0 if size != 2 else (FORK_REQ_R if side == 'L' else FORK_REQ_L))
        print(f'{INDENT_SIZE*index*"    "}trazim {side} vilicu {index}', flush=True)

    while forks[side] is None:
        # ako te netko trazi drugu vilicu u meduvremenu
        if comm.iprobe(source=neighbors[other], tag=0 if size != 2 else (FORK_REQ_L if side == 'L' else FORK_REQ_R)):
            check_neighbor_request(other,index,size)

        # ako ti posalje trazenu vilicu
        if comm.iprobe(source=neighbors[side], tag=0 if size != 2 else (FORK_SEND_R if side == 'L' else FORK_SEND_L)):
            msg = comm.recv(source=neighbors[side], tag=0 if size != 2 else (FORK_SEND_R if side == 'L' else FORK_SEND_L))
            forks[side] = 'C'


def process(index, size):
    
    # random.seed(getpid())
    
    while True:

        # misli
        rand_think = random.randint(2, 5)
        print(f'{INDENT_SIZE*index*"    "}mislim {index}', flush=True)
        for _ in range(rand_think):
            sleep(1)
            check_requests(index, size)

        # provjeri vilice
        while not forks['L'] or not forks['R']:
            for side in ['L', 'R']:
                check_forks(side, index, size)

        # jedi
        print(f'{INDENT_SIZE*index*"    "}jedem {index}', flush=True)
        sleep(1)
        forks['L'] = 'D'
        forks['R'] = 'D'

        # proslijedi vilice ako su te trazili
        for side in ['L', 'R']:
            if send[side]:
                send[side] = False
                forks[side] = None
                print(f'{INDENT_SIZE*index*"    "}{send}')
                comm.send(True, dest=neighbors[side], tag=0 if size != 2 else (FORK_SEND_L if side == 'L' else FORK_SEND_R))


def main():
    index = comm.Get_rank()
    size = comm.Get_size()

    set_neighbors(index, size)
    set_forks(index, size)
    process(index, size)

main()