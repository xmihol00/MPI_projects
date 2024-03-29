import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

success = True
wrong_bytes = []
with open(file1, 'rb') as f1, open(file2, 'r') as f2:
    binary = f1.read()
    text = f2.read().strip()
    text = text.split(' ')
    text = [int(x) for x in text]

    for i, (binary_byte, text_byte) in enumerate(zip(binary, text)):
        if binary_byte != text_byte:
            success = False
            wrong_bytes.append((i, binary_byte, text_byte))

if success:
    print("\033[92mSAME INPUTS\033[0m")
    exit(0)
else:
    print("\033[91mDIFFERENT INPUTS\033[0m")
    print(wrong_bytes)
    exit(1)
