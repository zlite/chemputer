import sys
import time

import board
import neopixel
import supervisor


NUM_PIXELS = 1
NEOPIXEL_PIN = board.GP16
BRIGHTNESS = 0.2
STEP_DELAY = 0.02
STEP_SIZE = 5


pixels = neopixel.NeoPixel(
    NEOPIXEL_PIN,
    NUM_PIXELS,
    brightness=BRIGHTNESS,
    auto_write=True,
)


def wheel(pos: int) -> tuple[int, int, int]:
    pos %= 256
    if pos < 85:
        return 255 - pos * 3, pos * 3, 0
    if pos < 170:
        pos -= 85
        return 0, 255 - pos * 3, pos * 3
    pos -= 170
    return pos * 3, 0, 255 - pos * 3


def clamp_byte(value: int) -> int:
    return max(0, min(255, value))


def parse_color(command: str):
    command = command.strip()
    if not command:
        return None

    lowered = command.lower()
    if lowered in ("help", "?"):
        return "help"
    if lowered in ("rainbow", "auto"):
        return "rainbow"
    if lowered in ("off", "black"):
        return (0, 0, 0)

    if command.startswith("#") and len(command) == 7:
        try:
            return (
                int(command[1:3], 16),
                int(command[3:5], 16),
                int(command[5:7], 16),
            )
        except ValueError:
            return None

    parts = [part.strip() for part in command.split(",")]
    if len(parts) == 3:
        try:
            return tuple(clamp_byte(int(part)) for part in parts)
        except ValueError:
            return None

    return None


def show_help():
    print("Send one of these commands, then press Enter:")
    print("  255,0,128   -> set RGB directly")
    print("  #FF0080     -> set RGB as hex")
    print("  rainbow     -> resume rainbow cycle")
    print("  off         -> turn LED off")


print("RP2040-Zero RGB serial control ready.")
show_help()

line_buffer = ""
mode = "rainbow"
current_color = (0, 0, 0)
rainbow_pos = 0


while True:
    while supervisor.runtime.serial_bytes_available:
        char = sys.stdin.read(1)
        if char in ("\r", "\n"):
            command = line_buffer.strip()
            line_buffer = ""
            if not command:
                continue

            result = parse_color(command)
            if result == "help":
                show_help()
            elif result == "rainbow":
                mode = "rainbow"
                print("Mode: rainbow")
            elif isinstance(result, tuple):
                mode = "solid"
                current_color = result
                pixels[0] = current_color
                print("Color set to {},{},{}".format(*current_color))
            else:
                print("Could not parse command:", command)
                show_help()
        else:
            line_buffer += char

    if mode == "rainbow":
        pixels[0] = wheel(rainbow_pos)
        rainbow_pos = (rainbow_pos + STEP_SIZE) % 256

    time.sleep(STEP_DELAY)
