import os


try:
    from colorama import Fore, Style, just_fix_windows_console

    just_fix_windows_console()
    COLORAMA_AVAILABLE = True
except Exception:
    COLORAMA_AVAILABLE = False

    class Fore:
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""

    class Style:
        BRIGHT = ""
        DIM = ""
        RESET_ALL = ""


COLOR_ENABLED = COLORAMA_AVAILABLE and "NO_COLOR" not in os.environ


def paint(text, color="", bright=False, dim=False):
    if not COLOR_ENABLED:
        return str(text)
    style = ""
    if bright:
        style += Style.BRIGHT
    if dim:
        style += Style.DIM
    return f"{style}{color}{text}{Style.RESET_ALL}"


def section(title):
    print(paint(f"[{title}]", Fore.CYAN, bright=True))


def success(text):
    print(paint(text, Fore.GREEN, bright=True))


def warning(text):
    print(paint(text, Fore.YELLOW, bright=True))


def error(text):
    print(paint(text, Fore.RED, bright=True))


def info(text):
    print(paint(text, Fore.CYAN))


def dim(text):
    print(paint(text, Fore.WHITE, dim=True))


def key_value(key, value, color=Fore.WHITE):
    print(f"  {paint(key + ':', color, bright=True)} {value}")


def separator(width=78):
    print(paint("=" * width, Fore.BLUE, bright=True))


def color_package_hint():
    if COLORAMA_AVAILABLE:
        return "colorama loaded"
    return "colorama not installed; run `pip install colorama` for colored Windows terminal output"
