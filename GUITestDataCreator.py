from tkinter import Tk, Button, LEFT, RIGHT, Label, CENTER, TOP, BOTTOM, Scrollbar, HORIZONTAL, VERTICAL, Y, X, NW
from tkinter import filedialog as fd
from tkinter.ttk import Frame, Treeview, Style
from tkinter import messagebox as mb

from PIL import ImageTk, Image
from src.TestDataCreator import TestDataCreator


class GUI:
    def __init__(self):
        self.root = Tk()
        self.right_frame = Frame(self.root)
        self.left_frame = Frame(self.root)
        self.image = None
        self.scrollbarx = Scrollbar(self.left_frame, orient=HORIZONTAL)
        self.scrollbary = Scrollbar(self.left_frame, orient=VERTICAL)
        self.tree = None
        self.files = {}
        self.tree_index = None
        self.selected_item = None

        self.test_data_creator = TestDataCreator()
        self.char_images = []
        self.index = 0
        self.panel = None

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
        self.add_buttons()

        self.left_frame.pack(side=LEFT)
        self.right_frame.pack(side=RIGHT)

    def add_buttons(self):
        self.panel = Label(self.right_frame, image=self.image)
        self.panel.pack(side=TOP)

        digit_frame = Frame(self.right_frame)
        for i in [str(i) for i in range(10)]:
            button_digit = Button(digit_frame, text=i, width=3)
            button_digit.bind('<Button-1>', self.button_click)
            button_digit.pack(side=LEFT, padx=2.)
        digit_frame.pack(pady=5)

        alpha_frame = Frame(self.right_frame)
        for i in ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X']:
            button_alpha = Button(alpha_frame, text=i, width=3)
            button_alpha.bind('<Button-1>', self.button_click)
            button_alpha.pack(side=LEFT, padx=2.)
        alpha_frame.pack(pady=5)

        button_nothing = Button(self.right_frame, text='-', width=10)
        button_nothing.bind('<Button-1>', self.button_click)
        button_nothing.pack()

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

    def button_click(self, event):
        image_name = event.widget.cget('text')
        if image_name != '-':
            self.test_data_creator.multiply_image(self.char_images[self.index], image_name)

        self.index += 1
        if self.index >= len(self.char_images):
            del self.files[self.selected_item]
            mb.showinfo("Recognition", "Characters are over!")
            self.image = None
            self.create_tree()
        else:
            self.image = ImageTk.PhotoImage(Image.fromarray(self.char_images[self.index].image).resize((120, 200)))
        self.panel.configure(image=self.image)

    def run(self):
        index = self.tree.selection()[0]

        if not index:
            mb.showinfo("Choice image", "Not selected item")
            return
        self.selected_item = ' '.join(self.tree.item(index)['values'])
        self.char_images = self.test_data_creator.start(self.selected_item)
        self.index = 0
        if not self.char_images:
            mb.showerror("Recognition", "Car numbers not recognized")
            del self.files[self.selected_item]
            self.create_tree()
            self.image = None
        else:
            self.image = ImageTk.PhotoImage(Image.fromarray(self.char_images[self.index].image).resize((120, 200)))
            self.panel.configure(image=self.image)


if __name__ == '__main__':
    gui = GUI()
    gui.create_root()
    gui.root.mainloop()
