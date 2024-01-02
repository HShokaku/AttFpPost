from typing import List

'''
This section of code was adapted from the Chemprop project
Original Chemprop code is under the MIT License:
https://github.com/chemprop/chemprop/blob/9bc0d0ef483bd6e43ab097bbb5b93a7b065f1fa2/chemprop/features/featurization.py
'''

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding