"""Convenience wrapper to launch the GUI from the workspace root.

You experienced an error when running `python gui.py` from the project root
because the real script lives under `src/gui.py`.  This file simply imports
and calls the launcher so that the command works as expected.
"""

from src.gui import SteelPredictorApp
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = SteelPredictorApp(root)
    root.mainloop()

