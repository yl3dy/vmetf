#!/usr/bin/env python3

import sys

def main():
    if len(sys.argv) != 2 or int(sys.argv[1]) not in range(1,6):
        print('Bad arguments. Should specify only method number (1-5)')
        quit()
    arg = sys.argv[1]
    prompt = lambda desc: 'Using method {}: {}'.format(arg, desc)
    if arg == '1':
        print(prompt(''))
    elif arg == '2':
        print(prompt(''))
    elif arg == '3':
        print(prompt(''))
    elif arg == '4':
        print(prompt(''))
    elif arg == '5':
        print(prompt(''))

if __name__ == '__main__':
    main()
