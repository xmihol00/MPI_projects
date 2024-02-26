import sys

direction = True
if sys.argv[1] == '-d':
    direction = False
size = int(sys.argv[2])

current_max = 0
current_min = 255
sorted = True

actual_size = 0
if direction:
    for line in sys.stdin:
        actual_size += 1
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
        actual_size += 1
        print(line, end='')
        line = line.strip()
        value = int(line)
        if value < current_min:
            current_min = value
        if value > current_min:
            print('Error: input is not sorted', file=sys.stderr)
            sorted = False

if actual_size == size:
    print('\033[92mSizes match\033[0m', file=sys.stderr)
else:
    print('\033[91mSizes do not match\033[0m', f'actual={actual_size}, expected={size}', file=sys.stderr)

if sorted:
    print('\033[92mInput is sorted\033[0m', file=sys.stderr)
else:
    print('\033[91mInput is not sorted\033[0m', file=sys.stderr)
        