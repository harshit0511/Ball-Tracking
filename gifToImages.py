import os
from PIL import Image

def gifToImages(gifName):
    frame = Image.open(gifName + ".gif")
    nframes = 0
    if not os.path.exists(gifName):
        os.makedirs(gifName)
    while frame:
        frame.save( "{}/{}-{}.png".format(gifName, os.path.basename(gifName), nframes ) , 'PNG')
        nframes += 1
        try:
            frame.seek( nframes )
        except EOFError:
            break;
    return True

gifToImages("square2")