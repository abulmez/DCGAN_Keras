#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.23a
#  in conjunction with Tcl version 8.6
#    Jun 22, 2019 05:28:37 PM EEST  platform: Windows NT
import ctypes
import threading

from service.DCGAN_trainer import DCGANTrainer

top_level = None

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True

from ui import DCGAN_UI_support


def fill_entries_with_default_data():
    top_level.epochs_entry.insert(0, "10")
    top_level.batch_size_entry.insert(0, "128")
    top_level.training_data_folder_name_entry.insert(0, "celebA_small")
    top_level.z_dim_entry.insert(0, "512")
    top_level.sampling_interval_entry.insert(0, "10")


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root, top_level
    root = tk.Tk()
    top = MainWindow(root)
    DCGAN_UI_support.init(root, top)
    top_level = top
    fill_entries_with_default_data()
    root.mainloop()


w = None


def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel(root)
    top = MainWindow(w)
    DCGAN_UI_support.init(w, top, *args, **kwargs)
    return (w, top)


def destroy_Toplevel1():
    global w
    w.destroy()
    w = None


class MainWindow:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'

        top.geometry("999x707+-2006+422")
        top.title("DCGAN")
        top.configure(background="#d9d9d9")

        self.start_button = tk.Button(top, command=start_training_callback)
        self.start_button.place(relx=0.09, rely=0.651, height=43, width=156)
        self.start_button.configure(activebackground="#ececec")
        self.start_button.configure(activeforeground="#000000")
        self.start_button.configure(background="#d9d9d9")
        self.start_button.configure(disabledforeground="#a3a3a3")
        self.start_button.configure(foreground="#000000")
        self.start_button.configure(highlightbackground="#d9d9d9")
        self.start_button.configure(highlightcolor="black")
        self.start_button.configure(pady="0")
        self.start_button.configure(text='''Start''')
        self.start_button.configure(width=156)

        self.ResultsCanvas = tk.Canvas(top)
        self.ResultsCanvas.place(relx=0.31, rely=0.042, relheight=0.909
                                 , relwidth=0.644)
        self.ResultsCanvas.configure(background="#d9d9d9")
        self.ResultsCanvas.configure(borderwidth="2")
        self.ResultsCanvas.configure(highlightbackground="#d9d9d9")
        self.ResultsCanvas.configure(highlightcolor="black")
        self.ResultsCanvas.configure(insertbackground="black")
        self.ResultsCanvas.configure(relief="ridge")
        self.ResultsCanvas.configure(selectbackground="#c4c4c4")
        self.ResultsCanvas.configure(selectforeground="black")
        self.ResultsCanvas.configure(width=640)

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=0.03, rely=0.099, height=26, width=53)
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Epochs''')

        self.epochs_entry = tk.Entry(top)
        self.epochs_entry.place(relx=0.03, rely=0.127, height=24, relwidth=0.144)

        self.epochs_entry.configure(background="white")
        self.epochs_entry.configure(disabledforeground="#a3a3a3")
        self.epochs_entry.configure(font="TkFixedFont")
        self.epochs_entry.configure(foreground="#000000")
        self.epochs_entry.configure(insertbackground="black")
        self.epochs_entry.configure(width=144)

        self.Label1_5 = tk.Label(top)
        self.Label1_5.place(relx=0.01, rely=0.184, height=26, width=102)
        self.Label1_5.configure(activebackground="#f9f9f9")
        self.Label1_5.configure(activeforeground="black")
        self.Label1_5.configure(background="#d9d9d9")
        self.Label1_5.configure(disabledforeground="#a3a3a3")
        self.Label1_5.configure(foreground="#000000")
        self.Label1_5.configure(highlightbackground="#d9d9d9")
        self.Label1_5.configure(highlightcolor="black")
        self.Label1_5.configure(text='''Batch size''')
        self.Label1_5.configure(width=102)

        self.batch_size_entry = tk.Entry(top)
        self.batch_size_entry.place(relx=0.03, rely=0.212, height=24
                                    , relwidth=0.144)
        self.batch_size_entry.configure(background="white")
        self.batch_size_entry.configure(disabledforeground="#a3a3a3")
        self.batch_size_entry.configure(font="TkFixedFont")
        self.batch_size_entry.configure(foreground="#000000")
        self.batch_size_entry.configure(highlightbackground="#d9d9d9")
        self.batch_size_entry.configure(highlightcolor="black")
        self.batch_size_entry.configure(insertbackground="black")
        self.batch_size_entry.configure(selectbackground="#c4c4c4")
        self.batch_size_entry.configure(selectforeground="black")
        self.batch_size_entry.configure(width=144)

        self.Label1_6 = tk.Label(top)
        self.Label1_6.place(relx=0.01, rely=0.255, height=46, width=220)
        self.Label1_6.configure(activebackground="#f9f9f9")
        self.Label1_6.configure(activeforeground="black")
        self.Label1_6.configure(background="#d9d9d9")
        self.Label1_6.configure(disabledforeground="#a3a3a3")
        self.Label1_6.configure(foreground="#000000")
        self.Label1_6.configure(highlightbackground="#d9d9d9")
        self.Label1_6.configure(highlightcolor="black")
        self.Label1_6.configure(text='''Training/testing data folder name''')
        self.Label1_6.configure(width=172)

        self.training_data_folder_name_entry = tk.Entry(top)
        self.training_data_folder_name_entry.place(relx=0.03, rely=0.297
                                                   , height=24, relwidth=0.144)
        self.training_data_folder_name_entry.configure(background="white")
        self.training_data_folder_name_entry.configure(disabledforeground="#a3a3a3")
        self.training_data_folder_name_entry.configure(font="TkFixedFont")
        self.training_data_folder_name_entry.configure(foreground="#000000")
        self.training_data_folder_name_entry.configure(highlightbackground="#d9d9d9")
        self.training_data_folder_name_entry.configure(highlightcolor="black")
        self.training_data_folder_name_entry.configure(insertbackground="black")
        self.training_data_folder_name_entry.configure(selectbackground="#c4c4c4")
        self.training_data_folder_name_entry.configure(selectforeground="black")
        self.training_data_folder_name_entry.configure(width=144)

        self.run_tests_checkbox_state = tk.IntVar()
        self.run_tests_checkbox = tk.Checkbutton(top, variable=self.run_tests_checkbox_state)
        self.run_tests_checkbox.place(relx=0.02, rely=0.537, relheight=0.044
                                      , relwidth=0.19)
        self.run_tests_checkbox.configure(activebackground="#ececec")
        self.run_tests_checkbox.configure(activeforeground="#000000")
        self.run_tests_checkbox.configure(background="#d9d9d9")
        self.run_tests_checkbox.configure(disabledforeground="#a3a3a3")
        self.run_tests_checkbox.configure(foreground="#000000")
        self.run_tests_checkbox.configure(highlightbackground="#d9d9d9")
        self.run_tests_checkbox.configure(highlightcolor="black")
        self.run_tests_checkbox.configure(justify='left')
        self.run_tests_checkbox.configure(text='''Run tests instead of training''')
        self.run_tests_checkbox.configure(variable=DCGAN_UI_support.che60)
        self.run_tests_checkbox.configure(width=98)

        self.z_dim_entry = tk.Entry(top)
        self.z_dim_entry.place(relx=0.03, rely=0.382, height=24, relwidth=0.144)
        self.z_dim_entry.configure(background="white")
        self.z_dim_entry.configure(disabledforeground="#a3a3a3")
        self.z_dim_entry.configure(font="TkFixedFont")
        self.z_dim_entry.configure(foreground="#000000")
        self.z_dim_entry.configure(insertbackground="black")
        self.z_dim_entry.configure(width=144)

        self.Label2 = tk.Label(top)
        self.Label2.place(relx=-0.01, rely=0.354, height=16, width=122)
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''Z-dim''')
        self.Label2.configure(width=122)

        self.Label2_8 = tk.Label(top)
        self.Label2_8.place(relx=0.01, rely=0.438, height=26, width=132)
        self.Label2_8.configure(activebackground="#f9f9f9")
        self.Label2_8.configure(activeforeground="black")
        self.Label2_8.configure(background="#d9d9d9")
        self.Label2_8.configure(disabledforeground="#a3a3a3")
        self.Label2_8.configure(foreground="#000000")
        self.Label2_8.configure(highlightbackground="#d9d9d9")
        self.Label2_8.configure(highlightcolor="black")
        self.Label2_8.configure(text='''Sampling Interval''')
        self.Label2_8.configure(width=132)

        self.sampling_interval_entry = tk.Entry(top)
        self.sampling_interval_entry.place(relx=0.03, rely=0.467, height=24
                                           , relwidth=0.144)
        self.sampling_interval_entry.configure(background="white")
        self.sampling_interval_entry.configure(cursor="fleur")
        self.sampling_interval_entry.configure(disabledforeground="#a3a3a3")
        self.sampling_interval_entry.configure(font="TkFixedFont")
        self.sampling_interval_entry.configure(foreground="#000000")
        self.sampling_interval_entry.configure(highlightbackground="#d9d9d9")
        self.sampling_interval_entry.configure(highlightcolor="black")
        self.sampling_interval_entry.configure(insertbackground="black")
        self.sampling_interval_entry.configure(selectbackground="#c4c4c4")
        self.sampling_interval_entry.configure(selectforeground="black")

        self.console_text_box = tk.Text(top)
        self.console_text_box.place(relx=0.02, rely=0.736, relheight=0.218
                                    , relwidth=0.284)
        self.console_text_box.configure(background="white")
        self.console_text_box.configure(font="TkTextFont")
        self.console_text_box.configure(foreground="black")
        self.console_text_box.configure(highlightbackground="#d9d9d9")
        self.console_text_box.configure(highlightcolor="black")
        self.console_text_box.configure(insertbackground="black")
        self.console_text_box.configure(selectbackground="#c4c4c4")
        self.console_text_box.configure(selectforeground="black")
        self.console_text_box.configure(width=284)
        self.console_text_box.configure(wrap="word")


class DCGANTrainerThread(threading.Thread):
    def __init__(self, ui):
        threading.Thread.__init__(self)
        self.ui = ui
        self.DCGANTrainer = None

    def run(self):
        self.DCGANTrainer = DCGANTrainer(int(self.ui.epochs_entry.get()), int(self.ui.batch_size_entry.get()),
                                         self.ui.training_data_folder_name_entry.get(),
                                         int(self.ui.z_dim_entry.get()),
                                         int(self.ui.sampling_interval_entry.get()),
                                         128, 128, self.ui.ResultsCanvas, self.ui.console_text_box)
        if self.ui.run_tests_checkbox_state.get():
            self.DCGANTrainer.run_tests()
        else:
            self.DCGANTrainer.train()

    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')


started = False
working_thread = None


def start_training_callback():
    global working_thread, started
    if started:
        # working_thread.raise_exception()
        working_thread.raise_exception()
        working_thread.join()
        started = False
        top_level.start_button.configure(text="Start")
    else:
        started = True
        working_thread = DCGANTrainerThread(top_level)
        working_thread.start()
        top_level.start_button.configure(text="Stop")


if __name__ == '__main__':
    vp_start_gui()
