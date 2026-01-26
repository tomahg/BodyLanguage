import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} word")
    sys.exit(1)

word = sys.argv[1]

for c in word:
    print(f"{c} -> {ord(c):3}")