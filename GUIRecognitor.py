from PIL import ImageTk, Image

from tkinter import Tk, Button, LEFT, RIGHT, Label, CENTER, TOP, BOTTOM, Scrollbar, HORIZONTAL, VERTICAL, Y, X, NW
from tkinter import filedialog as fd
from tkinter.ttk import Frame, Treeview, Style
from tkinter import messagebox as mb

from src.recognition import RecognitionCarPlate


class GUI:
    def __init__(self):
        self.root = Tk()
        self.right_frame = Frame(self.root)
        self.left_frame = Frame(self.root)
        self.scrollbarx = Scrollbar(self.left_frame, orient=HORIZONTAL)
        self.scrollbary = Scrollbar(self.left_frame, orient=VERTICAL)
        self.tree = None
        self.files = {}
        self.tree_index = None
        self.selected_item = None

        self.recogniser = RecognitionCarPlate()
        self.label = None
        self.panel = None
        self.image = None

        self.buttons = {}

    def create_root(self):
        self.root.title('Data Creator')
        w = 720
        h = 450
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) / 2
        y = (sh - h) / 2
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.root.resizable(False, False)
        self.add_left_frame()
        self.add_right_frame()

        self.left_frame.pack(side=LEFT)
        self.right_frame.pack(side=RIGHT)

    def add_right_frame(self):
        self.panel = Label(self.right_frame, image=self.image)
        self.panel.pack(side=LEFT, anchor=CENTER)

        self.label = Label(self.right_frame, text='RESULT')
        self.label.pack()

    def create_tree(self):
        if self.tree:
            self.tree.destroy()

        style = Style(self.left_frame)
        style.configure('Calendar.Treeview', rowheight=50)
        self.tree = Treeview(self.left_frame, columns='#1',
                             height=400, selectmode="extended",
                             yscrollcommand=self.scrollbary.set,
                             xscrollcommand=self.scrollbarx.set,
                             style='Calendar.Treeview')

        self.scrollbary.config(command=self.tree.yview)
        self.scrollbary.pack(side=RIGHT, fill=Y)
        self.scrollbarx.config(command=self.tree.xview)
        self.scrollbarx.pack(side=BOTTOM, fill=X)

        self.tree.heading('#0', text='image', anchor=CENTER)
        self.tree.heading('#1', text='file_name')

        self.tree.column('#0', width=80, anchor=CENTER)
        if self.files:
            for key in self.files:
                self.tree.insert("", 'end', values=key, image=self.files[key])
        self.tree.pack(side=BOTTOM)

    def add_left_frame(self):
        load_button = Button(self.left_frame, text="load files", command=self.load_files)
        load_button.pack(side=TOP, anchor=NW, pady=5)
        start_button = Button(self.left_frame, text='Start recognition', command=self.run)
        start_button.pack(side=TOP, anchor=NW)
        self.create_tree()

    def load_files(self):
        file_names = fd.askopenfilenames(filetypes=[('Image file', '*.png'), ('Image file', '*.jpg'),
                                                    ('Image file', '*.jpeg')])
        for file in file_names:
            self.files[file] = ImageTk.PhotoImage(Image.open(file).resize((60, 40)))
        self.create_tree()

    def run(self):
        index = self.tree.selection()

        if not index:
            mb.showinfo("Choice image", "Not selected item")
            return
        index = index[0]
        self.selected_item = ' '.join(self.tree.item(index)['values'])
        self.image = ImageTk.PhotoImage(Image.open(self.selected_item).resize((300, 200)))
        self.panel.configure(image=self.image)

        results = self.recogniser.run(self.selected_item)
        print(results)
        if not results:
            self.label['text'] = 'RESULT:\nNo results'
            mb.showinfo("Recognition", "Car number not recognized!")
        else:
            res = 'RESULT:\n'
            for i in results:
                for char in i:
                    res += char
                res += '\n'
            self.label['text'] = res


if __name__ == '__main__':
    gui = GUI()
    gui.create_root()
    gui.root.mainloop()