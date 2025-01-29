import os
from restack_ai.function import function, log
from pydantic import BaseModel

class BrailleInput(BaseModel):
    text: str

class BrailleOutput(BaseModel):
    braille_text: str

# Braille Conversion Function
def convert_text_to_braille(text):
    """
    Convert a given text into a Braille representation while preserving formatting,
    such as line breaks and paragraph spacing.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Invalid text input. Please provide non-empty string data.")

    braille_map = {
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
        'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
        'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
        'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
        'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
        'z': '⠵',
        'A': '⠠⠁', 'B': '⠠⠃', 'C': '⠠⠉', 'D': '⠠⠙', 'E': '⠠⠑',
        'F': '⠠⠋', 'G': '⠠⠛', 'H': '⠠⠓', 'I': '⠠⠊', 'J': '⠠⠚',
        'K': '⠠⠅', 'L': '⠠⠇', 'M': '⠠⠍', 'N': '⠠⠝', 'O': '⠠⠕',
        'P': '⠠⠏', 'Q': '⠠⠟', 'R': '⠠⠗', 'S': '⠠⠎', 'T': '⠠⠞',
        'U': '⠠⠥', 'V': '⠠⠧', 'W': '⠠⠺', 'X': '⠠⠭', 'Y': '⠠⠽',
        'Z': '⠠⠵',
        '0': '⠼⠚', '1': '⠼⠁', '2': '⠼⠃', '3': '⠼⠉', '4': '⠼⠙',
        '5': '⠼⠑', '6': '⠼⠋', '7': '⠼⠛', '8': '⠼⠓', '9': '⠼⠊',
        ',': '⠂', ';': '⠆', ':': '⠒', '.': '⠲', '!': '⠖',
        '(': '⠶', ')': '⠶', '?': '⠦', '-': '⠤', ' ': ' ',
        '\'': '⠄', '\"': '⠐', '/': '⠌', '\\': '⠸', '@': '⠈',
        '#': '⠼', '$': '⠫', '%': '⠩', '&': '⠯', '*': '⠡',
        '+': '⠬', '=': '⠿', '<': '⠣', '>': '⠜', '^': '⠘',
        '_': '⠸', '`': '⠈', '{': '⠷', '}': '⠾', '[': '⠪',
        ']': '⠻', '|': '⠳', '~': '⠴'
    }

    lines = text.splitlines()
    braille_lines = ["".join(braille_map.get(char, char) for char in line) for line in lines]

    return "\n".join(braille_lines)

@function.defn()
async def text_to_braille(input: BrailleInput) -> BrailleOutput:
    """
    Convert input text to Braille and return the result.
    """
    try:
        braille_text = convert_text_to_braille(input.text)
        log.info(f"Converted text to Braille: {braille_text}")
        return BrailleOutput(braille_text=braille_text)
    except Exception as e:
        log.error("Error in text_to_braille function", error=e)
        raise e
