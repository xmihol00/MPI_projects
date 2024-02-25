import sys

direction = True
if len(sys.argv) > 1:
    if sys.argv[1] == '-d':
        direction = False

current_max = 0
current_min = 255
sorted = True

if direction:
    for line in sys.stdin:
        print(line, end='')
        line = line.strip()
        value = int(line)
        if value > current_max:
            current_max = value
        if value < current_max:
            print('Error: input is not sorted', file=sys.stderr)
            sorted = False
else:
    for line in sys.stdin:
        print(line, end='')
        line = line.strip()
        value = int(line)
        if value < current_min:
            current_min = value
        if value > current_min:
            print('Error: input is not sorted', file=sys.stderr)
            sorted = False

if sorted:
    print('Input is sorted', file=sys.stderr)
else:
    print('Input is not sorted', file=sys.stderr)
        