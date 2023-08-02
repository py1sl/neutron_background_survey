import os
import src.diamon as d

MAIN_MENU = """
**************************************
DIAMON Spectrometer read & analyse data
***************************************
 <*> Open DIAMON data (open)
 <*> Analyse data (analyse)
 <*> Exit (end)
"""
READ_MENU = """
*********************************
Select diamon files to read
*********************************
 <*> all data (all)
 <*> TS1 (ts1)
 <*> TS2 (ts2)
 <*> TS1 Instrument (inst_1)
 <*> TS2 Instrument (inst_2)
"""
def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
class menu:
    def __init__(self):
        self.main_menu()
    def main_menu(self):
        #clear_screen()
        print(MAIN_MENU)
        command = input("Enter an action from above: ")
        self.process_input(command)

    def process_input(self, command):
        if command == 'open':
            return self.command_open()
        elif command == 'analyse':
            return self.command_analyse()
        elif command == 'end':
            print("\n Goodbye!")
        else:
            print("invalid command, try again: ")
            self.main_menu()
    def process_read(self, command):
        return
    def open_menu(self, folder):
        print(READ_MENU)
        command = input("Enter which data you want: ")
        self.process_read(command)
    def command_open(self):
        folder = input("Read data from TS1: ts1 or TS2: ts2 or all: all")
        try:
            diamon_data = d.diamon(folder)
        except FileNotFoundError:
            print("no file found!")
            return self.main_menu()
        return self.main_menu()
    def command_analyse(self):
        return
    def command_exit(self):
        return