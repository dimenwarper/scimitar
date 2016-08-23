import matplotlib.colors as colors
STATE_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                'gray', 'tan', 'brown', 'salmon']
_more_colors = [c for c in colors.cnames if c not in STATE_COLORS]
STATE_COLORS += _more_colors
