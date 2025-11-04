# CMPUT 455 Assignment 3 starter code (PoE2)
# Implement the specified commands to complete the assignment
# Full assignment specification on Canvas

import sys
import signal
import re

class CommandInterface:
    # The following is already defined and does not need modification
    # However, you may change or add to this code as you see fit, e.g. adding class variables to init

    def __init__(self):
        # Define the string to function command mapping
        self.command_dict = {
            "help"     : self.help,
            "init_game": self.init_game,   # init_game w h p s [board]
            "show"     : self.show,
            "timelimit": self.timelimit,   # timelimit seconds
            "load_patterns"    : self.load_patterns,       # see assignment spec
            "policy_moves"    : self.policy_moves,       # see assignment spec
            "position_evaluation": self.position_evaluation, # see assignment spec
            "move_evaluation": self.move_evaluation, # see assignment spec
            "score"    : self.score,
            "play" : self.play,
            "undo" : self.undo  # Add undo support
        }
        
        # Game state
        self.board = [[None]]
        self.player = 1           # 1 or 2
        self.handicap = 0.0       # P2’s handicap
        self.score_cutoff = float("inf")

        # This variable keeps track of the maximum allowed time to solve a position
        self.timelimit = 1
        
        # Patterns will be compiled dicts:
        # {"pat": str, "rev": str|None, "len": int, "w": float, "rx": regex, "rx_rev": regex|None}
        self.patterns = []

    # Convert a raw string to a command and a list of arguments
    def process_command(self, s):
        
        class TimeoutException(Exception):
            pass
        
        def handler(signum, frame):
            raise TimeoutException("Function timed out.")
        
        s = s.lower().strip()
        if len(s) == 0:
            return True
        command = s.split(" ")[0]
        args = [x for x in s.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Uknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timelimit)
            
            return self.command_dict[command](args)
        except TimeoutException:
            print(f"Command '{s}' timed out after {self.timelimit} seconds.", file=sys.stderr)
            print("= -1\n")
            return False
        except Exception as e:
            print("Command '" + s + "' failed with exception:", file=sys.stderr)
            print(e, file=sys.stderr)
            print("= -1\n")
            return False
        finally: 
            signal.alarm(0)

        
    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            s = input()
            if s.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(s):
                print("= 1\n")

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    # Helper function for command argument checking
    # Will make sure there are enough arguments, and that they are valid integers
    def arg_check(self, args, template):
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Recieved arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                args[i] = int(arg)
            except ValueError:
                try:
                    args[i] = float(arg)
                except ValueError:
                    print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr)
                    return False
        return True


    # init_game w h p s [board_str]
    # Note that your init_game function must support initializing the game with a string (this was not necessary in A1).
    # We already have implemented this functionality in our provided init_game function.
    def init_game(self, args):
        # Check arguments
        if len(args) > 4:
            self.board_str = args.pop()
        else:
            self.board_str = ""
        if not self.arg_check(args, "w h p s"):
            return False
        w, h, p, s = args
        if not (1 <= w <= 10000 and 1 <= h <= 10000):
            print("Invalid board size:", w, h, file=sys.stderr)
            return False
        
        #Initialize game state

        self.width = w
        self.height = h
        self.handicap = p
        if s == 0:
            self.score_cutoff = float("inf")
        else:
            self.score_cutoff = s
        
        self.board = []
        for r in range(self.height):
            self.board.append([0]*self.width)
        self.player = 1
        self.move_history = []

        # optional board string to initialize the game state
        if len(self.board_str) > 0:
            board_rows = self.board_str.split("/")
            if len(board_rows) != self.height:
                print("Board string has wrong height.", file=sys.stderr)
                return False
            
            p1_count = 0
            p2_count = 0
            for y, row_str in enumerate(board_rows):
                if len(row_str) != self.width:
                    print("Board string has wrong width.", file=sys.stderr)
                    return False
                for x, c in enumerate(row_str):
                    if c == "1":
                        self.board[y][x] = 1
                        p1_count += 1
                    elif c == "2":
                        self.board[y][x] = 2
                        p2_count += 1
            
            if p1_count > p2_count:
                self.player = 2
            else:
                self.player = 1

        self.timelimit = 1

        return True
    
    def show(self, args):
        for row in self.board:
            print(" ".join(["_" if v == 0 else str(v) for v in row]))
        return True

    def timelimit(self, args):
        """
        >> timelimit <seconds>
        Sets the wall-clock time limit used by 'solve'.
        - Accepts a single non-negative integer.
        """
        if not self.arg_check(args, "s"):
            return False

        self.timelimit = int(args[0])
        return True
    
    # Adding undo functionality
    def undo(self, args):
        if not self.move_history:
            return False
        
        x, y = self.move_history.pop()
        self.board[y][x] = 0
        self.player = 2 if self.player == 1 else 1
        return True
    
    # The following functions do not need to be callable as commands in assignment 2, but implement the PoE2 game environment for you.
    # Feel free to change or modify or replace as needed, your implementation of A1 may provide better optimized methods.
    # These functions work, but are not necessarily computationally efficient.
    # There are different approaches to exploring state spaces, this starter code provides one approach, but you are not required to use these functions.

    def get_moves(self):
        moves = []
        for y in range(self.height):
            row = self.board[y]
            for x in range(self.width):
                if row[x] == 0:
                    moves.append((x, y))
        return moves

    def make_move(self, x, y):
        self.board[y][x] = self.player
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
            
    def undo_move(self, x, y):
        self.board[y][x] = 0
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    # Returns p1_score, p2_score
    def calculate_score(self):
        p1_score = 0
        p2_score = self.handicap

        # Progress from left-to-right, top-to-bottom
        # We define lines to start at the topmost (and for horizontal lines leftmost) point of that line
        # At each point, score the lines which start at that point
        # By only scoring the starting points of lines, we never score line subsets
        for y in range(self.height):
            for x in range(self.width):
                c = self.board[y][x]
                if c != 0:
                    lone_piece = True # Keep track of the special case of a lone piece
                    # Horizontal
                    hl = 1
                    if x == 0 or self.board[y][x-1] != c: #Check if this is the start of a horizontal line
                        x1 = x+1
                        while x1 < self.width and self.board[y][x1] == c: #Count to the end
                            hl += 1
                            x1 += 1
                    else:
                        lone_piece = False
                    # Vertical
                    vl = 1
                    if y == 0 or self.board[y-1][x] != c: #Check if this is the start of a vertical line
                        y1 = y+1
                        while y1 < self.height and self.board[y1][x] == c: #Count to the end
                            vl += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Diagonal
                    dl = 1
                    if y == 0 or x == 0 or self.board[y-1][x-1] != c: #Check if this is the start of a diagonal line
                        x1 = x+1
                        y1 = y+1
                        while x1 < self.width and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            dl += 1
                            x1 += 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Anit-diagonal
                    al = 1
                    if y == 0 or x == self.width-1 or self.board[y-1][x+1] != c: #Check if this is the start of an anti-diagonal line
                        x1 = x-1
                        y1 = y+1
                        while x1 >= 0 and y1 < self.height and self.board[y1][x1] == c: #Count to the end
                            al += 1
                            x1 -= 1
                            y1 += 1
                    else:
                        lone_piece = False
                    # Add scores for found lines
                    for line_length in [hl, vl, dl, al]:
                        if line_length > 1:
                            if c == 1:
                                p1_score += 2 ** (line_length-1)
                            else:
                                p2_score += 2 ** (line_length-1)
                    # If all found lines are length 1, check if it is the special case of a lone piece
                    if hl == vl == dl == al == 1 and lone_piece:
                        if c == 1:
                            p1_score += 1
                        else:
                            p2_score += 1

        return p1_score, p2_score
    
    def score(self, args):
        scores = self.calculate_score()
        print(f"{scores[0]} {scores[1]}")
        return True
    
    # Returns is_terminal, winner
    # Assumes no draws
    def is_terminal(self):
        p1_score, p2_score = self.calculate_score()
        if p1_score >= self.score_cutoff:
            return True, 1
        elif p2_score >= self.score_cutoff:
            return True, 2
        else:
            # Check if the board is full
            for y in range(self.height):
                for x in range(self.width):
                    if self.board[y][x] == 0:
                        return False, 0
            # The board is full, assign the win to the greater scoring player
            if p1_score > p2_score:
                return True, 1
            else:
                return True, 2
    
    def is_pos_avail(self, c, r):
        return self.board[r][c] == 0        
    
    # this function may be modified as needed, but should behave as expected
    def play(self, args):
        '''
            >> play <col> <row>
            Places current player's piece at position (<col>, <row>).
        '''
        if not self.arg_check(args, "x y"):
            return False
        
        try:
            col = int(args[0])
            row = int(args[1])
        except ValueError:
            return False
        
        if col < 0 or col >= len(self.board[0]) or row < 0 or row >= len(self.board) or not self.is_pos_avail(col, row):
            return False
            
        scores = self.calculate_score()
        if scores[0] >= self.score_cutoff or scores[1] >= self.score_cutoff:
            return False

        # put the piece onto the board
        self.board[row][col] = self.player
        self.move_history.append((col, row))

        # switch player
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
        
        return True

    # ---------------- Assignment 3: Pattern Matching Core ----------------

    def load_patterns(self, args):
        if len(args) < 1:
            return False
        filepath = args[0]
        try:
            compiled = []
            with open(filepath, 'r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    pat = parts[0].upper()
                    w = float(parts[1])

                    # compile original and reversed (if different)
                    def to_regex(p):
                        # '*' matches P/O/_ ONLY (NOT X)
                        rx = ''.join('[PO_]' if c == '*' else re.escape(c) for c in p)
                        # lookahead to capture overlapping matches
                        return re.compile(f"(?=({rx}))")

                    rev = pat[::-1]
                    compiled.append({
                        "pat": pat,
                        "rev": rev if rev != pat else None,
                        "len": len(pat),
                        "w": w,
                        "rx": to_regex(pat),
                        "rx_rev": (to_regex(rev) if rev != pat else None),
                    })
            self.patterns = compiled
            return True
        except Exception:
            return False

    # -------- Pattern semantics helpers --------

    def _board_char_at(self, x, y):
        """
        Return:
          'X' if (x,y) is off-board
          'P' if current-player stone at (x,y)
          'O' if opponent stone at (x,y)
          '_' if empty on-board
        """
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return 'X'
        v = self.board[y][x]
        if v == 0:
            return '_'
        return 'P' if v == self.player else 'O'

    def _map_cell_char(self, x, y):
        # Fast mapping to 'P','O','_' for on-board cells (no 'X' here)
        v = self.board[y][x]
        if v == 0:
            return '_'
        return 'P' if v == self.player else 'O'

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _line_starts(self, dx, dy):
        """
        Yield all (x0,y0) starts for maximal lines in direction (dx,dy):
        a start has the property that (x0-dx, y0-dy) is out of bounds.
        """
        W, H = self.width, self.height
        starts = []
        # along top/bottom depending on dy
        for x in range(W):
            y = 0 if dy >= 0 else H - 1
            x_prev = x - dx
            y_prev = y - dy
            if not self._in_bounds(x_prev, y_prev):
                if self._in_bounds(x, y):
                    starts.append((x, y))
        # along left/right depending on dx
        for y in range(H):
            x = 0 if dx >= 0 else W - 1
            x_prev = x - dx
            y_prev = y - dy
            if not self._in_bounds(x_prev, y_prev):
                if self._in_bounds(x, y):
                    starts.append((x, y))
        # de-duplicate while preserving order
        seen = set()
        out = []
        for s in starts:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _build_lines(self, Lmax):
        """
        Build all 4 base-direction lines, with X-padding of length Lmax-1 at both ends.
        Returns a list of dicts: { "coords": coords_list_with_offboard, "s": string_with_padding }.
        The coords list includes off-board coordinates for padding (so coverage includes X positions).
        """
        if Lmax < 1:
            return []

        lines = []
        directions = [(1,0), (0,1), (1,1), (-1,1)]
        pad = 'X' * (Lmax - 1)

        for dx, dy in directions:
            for sx, sy in self._line_starts(dx, dy):
                # collect on-board coords for this maximal line
                coords_on = []
                x, y = sx, sy
                while self._in_bounds(x, y):
                    coords_on.append((x, y))
                    x += dx
                    y += dy
                if not coords_on:
                    continue

                # on-board char string for current player
                chars = ''.join(self._map_cell_char(cx, cy) for (cx, cy) in coords_on)

                # build padding coordinates (off-board) before and after
                pre_coords = []
                px, py = coords_on[0][0] - dx, coords_on[0][1] - dy
                for _ in range(Lmax - 1):
                    pre_coords.append((px, py))
                    px -= dx
                    py -= dy
                pre_coords.reverse()

                post_coords = []
                qx, qy = coords_on[-1][0] + dx, coords_on[-1][1] + dy
                for _ in range(Lmax - 1):
                    post_coords.append((qx, qy))
                    qx += dx
                    qy += dy

                full_coords = pre_coords + coords_on + post_coords
                full_str = pad + chars + pad
                lines.append({"coords": full_coords, "s": full_str})
        return lines

    def get_all_matches(self):
        """
        Use precompiled regex on padded lines to find all matches quickly.
        Each match yields coverage coords INCLUDING off-board (for X).
        Returns list of (tuple(coords), weight).  (Tuple for cheap hashing/dedup.)
        """
        matches = []
        if not self.patterns or self.width == 0 or self.height == 0:
            return matches

        Lmax = max((p["len"] for p in self.patterns), default=1)
        lines = self._build_lines(Lmax)

        # For each pattern & each line, run regex and collect matches
        for p in self.patterns:
            L = p["len"]
            w = p["w"]
            rx = p["rx"]
            rx_rev = p["rx_rev"]

            for line in lines:
                s = line["s"]
                coords = line["coords"]

                # original pattern
                for m in rx.finditer(s):
                    i = m.start(1)
                    cov = tuple(coords[i:i+L])  # tuple (cheap)
                    matches.append((cov, w))

                # reversed pattern (if distinct)
                if rx_rev is not None:
                    for m in rx_rev.finditer(s):
                        i = m.start(1)
                        cov = tuple(coords[i:i+L])  # tuple (cheap)
                        matches.append((cov, w))

        return matches

    def filter_subset_matches(self, matches):
        """
        Priority:
          1) For identical covered coords (including off-board path cells), keep only the max weight.
          2) Drop any match that is a strict subset of a kept match.
        Uses tuple keys for fast dedup; converts to sets only for the kept items.
        """
        if not matches:
            return []

        # collapse identical coverage -> keep max weight
        best_by_coords = {}
        for cov_tuple, w in matches:
            prev = best_by_coords.get(cov_tuple)
            if (prev is None) or (w > prev):
                best_by_coords[cov_tuple] = w

        # sort by larger coverage first, then higher weight (both desc)
        items = sorted(best_by_coords.items(),
                       key=lambda kv: (len(kv[0]), kv[1]),
                       reverse=True)

        kept = []
        kept_sets = []  # list[set]
        for cov_tuple, w in items:
            cset = set(cov_tuple)  # convert once here
            # strict subset of any already-kept set?
            if any(cset < big for big in kept_sets):
                continue
            kept.append((cset, w))
            kept_sets.append(cset)

        return kept

    # ---------------- Evaluation Commands ----------------

    def position_evaluation(self, args):
        matches = self.get_all_matches()
        filtered = self.filter_subset_matches(matches)
        total = sum(w for _, w in filtered)
        print(f"{total:.1f}")
        return True

    def move_evaluation(self, args):
        moves = self.get_moves()
        vals = []

        for x, y in moves:
            self.make_move(x, y)  # player flips
            # evaluate child position for side-to-move in child
            matches = self.get_all_matches()
            filtered = self.filter_subset_matches(matches)
            score = sum(w for _, w in filtered)
            vals.append(-score)   # negamax
            self.undo_move(x, y)  # flip back

        # format: exact zeros as "0", others with one decimal
        out = []
        for v in vals:
            rv = round(v, 1)
            if rv == 0.0:
                out.append("0")
            else:
                out.append(f"{rv:.1f}")
        print(" ".join(out))
        return True

    def policy_moves(self, args):
        moves = self.get_moves()
        vals = []

        for x, y in moves:
            self.make_move(x, y)
            matches = self.get_all_matches()
            filtered = self.filter_subset_matches(matches)
            score = sum(w for _, w in filtered)
            vals.append(-score)
            self.undo_move(x, y)

        if not vals:
            print("")
            return True

        mmin = min(vals)
        shifted = [v - mmin + 1.0 for v in vals]
        denom = sum(shifted)
        probs = ([1.0 / len(vals)] * len(vals)) if denom == 0.0 else [s / denom for s in shifted]
        print(" ".join(f"{round(p, 3):.3f}" for p in probs))
        return True


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
