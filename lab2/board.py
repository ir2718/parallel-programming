EMPTY = 0

class Board:

    def __init__(self, rows=6, cols=7, last_mover=EMPTY, last_col=-1):
        self.rows = rows
        self.cols = cols
        self.last_mover = last_mover
        self.last_col = last_col
        self.field = [[EMPTY for j in range(self.cols)] for i in range(self.rows)]
        self.height = [0 for _ in range(self.cols)]

    def __str__(self):
        return '\n'.join([str(row) for row in reversed(self.field)]) + '\n'

    def move_legal(self, col):
        assert col <= self.cols
        return self.field[self.rows - 1][col] == EMPTY

    def move(self, col, player):
        if not self.move_legal(col):
            return False
        
        self.field[self.height[col]][col] = player
        self.height[col] += 1
        self.last_mover = player
        self.last_col = col
        return True
    
    def undo_move(self, col):
        assert col <= self.cols
        if self.height[col] == 0:
            return False
        
        self.field[self.height[col] - 1][col] = EMPTY
        self.height[col] -= 1
        return True
    
    def game_end(self):
        last_col = self.last_col
        assert last_col <= self.cols
        col = last_col
        row = self.height[last_col] - 1
        if row < 0:
            return False
        player = self.field[row][col]
        
        # uspravno
        seq = 1
        r = row - 1
        while (r >= 0) and (self.field[r][col] == player):
            seq += 1
            r -= 1
        
        if seq > 3:
            return True

        # vodoravno
        seq = 0
        c = col
        while ((c-1) >= 0) and (self.field[row][c-1] == player):
            c -= 1

        while (c < self.cols) and (self.field[row][c] == player):
            seq += 1
            c += 1
        
        if seq > 3:
            return True

        # koso s lijeva na desno
        seq = 0 
        r = row
        c = col
        while ((c-1) >= 0) and ((r-1) >= 0) and (self.field[r-1][c-1] == player):
            c -= 1
            r -= 1

        while (c < self.cols) and (r < self.rows) and (self.field[r][c] == player):
            seq += 1
            c += 1
            r += 1 
        
        if seq > 3:
            return True

        # koso s desno na lijevo
        seq = 0 
        r = row
        c = col
        while ((c-1) >= 0) and ((r+1) < self.rows) and (self.field[r+1][c-1] == player):
            c -= 1
            r += 1

        while (c < self.cols) and (r >= 0) and (self.field[r][c] == player):
            c += 1
            r -= 1 
            seq += 1
        
        if seq > 3:
            return True

        return False

CPU = 1
HUMAN = 2

# b = Board()
# b.move(4, CPU)

# b.move(3, HUMAN)
# b.move(3, CPU)

# b.move(2, HUMAN)
# b.move(2, HUMAN)
# b.move(2, CPU)

# b.move(1, HUMAN)
# b.move(1, HUMAN)
# b.move(1, HUMAN)
# b.move(1, CPU)
# print(b.game_end())
