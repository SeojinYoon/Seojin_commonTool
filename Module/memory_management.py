
def convert_byte(byte, unit = "kB"):
    div = 1024
    if unit == "B":
        div = 1 # One alphabet or number
    elif unit == "kB":
        div **= 1 # a few paragraphs
    elif unit == "MB":
        div **= 2 # 1 minute long song
    elif unit == "GB":
        div **= 3 # 30 minutes long HD movie
    elif unit == "TB":
        div **= 4 # 200 FHD movies
    return byte / div

