import sys

def print_sameline(string):
    sys.stdout.write('\033[K')
    print string, '\r'
