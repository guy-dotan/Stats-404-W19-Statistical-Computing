'''
    File name: tic_tac_toe.py
    Author: Guy Dotan
    Date: 01/21/2019
    Course: UCLA Stats 404
    Description: Basic tic-tac-toe game with randomized move inputs.
'''

import random

## - dictionary to create board and occupied spaces - ##
theBoard = {'top-L': ' ', 'top-M': ' ', 'top-R': ' ',
            'mid-L': ' ', 'mid-M': ' ', 'mid-R': ' ',
            'low-L': ' ', 'low-M': ' ', 'low-R': ' '
            }

## - method prints the current status of the game board - ##
def printBoard(board):
    print(board['top-L'] + '|' + board['top-M'] + '|' + board['top-R'])
    print('-+-+-')
    print(board['mid-L'] + '|' + board['mid-M'] + '|' + board['mid-R'])
    print('-+-+-')
    print(board['low-L'] + '|' + board['low-M'] + '|' + board['low-R'])

## - method that returns a list of all the keys that have an empty value in the board dictionary
def get_empty_positions(board):
    return [key for (key, value) in board.items() if value == ' ']

## - method to determine the winner based on the 8 possible winning scenarios - ##
def isWinner(board):
    if theBoard['top-L'] != ' ' and theBoard['top-L'] == theBoard['top-M'] and theBoard['top-M'] == theBoard['top-R']:
        return True
    elif theBoard['mid-L'] != ' ' and theBoard['mid-L'] == theBoard['mid-M'] and theBoard['mid-M'] == theBoard['mid-R']:
        return True
    elif theBoard['low-L'] != ' ' and theBoard['low-L'] == theBoard['low-M'] and theBoard['low-M'] == theBoard['low-R']:
        return True
    elif theBoard['top-L'] != ' ' and theBoard['top-L'] == theBoard['mid-L'] and theBoard['mid-L'] == theBoard['low-L']:
        return True
    elif theBoard['top-M'] != ' ' and theBoard['top-M'] == theBoard['mid-M'] and theBoard['mid-M'] == theBoard['low-M']:
        return True
    elif theBoard['top-R'] != ' ' and theBoard['top-R'] == theBoard['mid-R'] and theBoard['mid-R'] == theBoard['low-R']:
        return True
    elif theBoard['top-L'] != ' ' and theBoard['top-L'] == theBoard['mid-M'] and theBoard['mid-M'] == theBoard['low-R']:
        return True
    elif theBoard['low-L'] != ' ' and theBoard['low-L'] == theBoard['mid-M'] and theBoard['mid-M'] == theBoard['top-R']:
        return True
    else:
        return False

## - Random seeds to test four possible game results - ##
#random.seed(1)  # X wins on last turn
#random.seed(3)  # O wins before last turn
#random.seed(4)  # X wins before last turn
#random.seed(7)  # Game ends in draw

## - Game begins with X going first - ##
turn = 'X'
printBoard(theBoard)

## - Loop continues until winner is determined - ##
while (isWinner(theBoard) == False):
    move_list = get_empty_positions(theBoard)
    print('Turn for ' + turn + '. Continue (y/n)?')
    cont = input()
    if (cont.lower() == 'y'):
        theMove = random.sample(move_list, 1)
        theBoard[theMove[0]] = turn
        if turn == 'X':
            turn = 'O'
        else:
            turn = 'X'
    elif (cont.lower() == 'n'):
        print('Game Over')
        break
    else:
        print('Invalid input')
    printBoard(theBoard)
    if (isWinner(theBoard) == True and turn == 'X'):
        print('Game Over. O wins!')
    elif (isWinner(theBoard) == True and turn == 'O'):
        print('Game Over. X wins!')
    elif (isWinner(theBoard) == False and len(get_empty_positions(theBoard)) == 0):
        print('Game ends in a draw!')
        break
