from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
import csv
import os
from tkinter import ttk
import time

from testanswer import predict_image

gui = Tk()

gui.title('COVID-19 DETECTOR')

gui.geometry('600x400')

progress_bar = ttk.Progressbar(orient='horizontal', length=600, mode='determinate')
progress_bar.grid(row=150, columnspan=3, pady=10)


def predict():

    path = [
        os.path.join(x)
        for x in os.listdir(pathv.get()) if x[0] != '.'
    ]

    # run prediction function annd obtain prediccted class index
    progress_bar['maximum'] = len(path)
    class_list = []
    for j in range(len(path)):
        progress_bar['value'] = j
        progress_bar.update()
        index = predict_image(pathv.get() + '/' + path[j])
        class_list.append(index)
        print("Predicted Class ", index, "image name", path[j])
    progress_bar['value'] = 0
    p_num = class_list.count(1)
    n_num = class_list.count(0)
    c_num = class_list.count(2)
    print(str(n_num) + '  ' + str(p_num) + '  ' + str(c_num))
    if p_num > 0.4 * len(class_list):
        pre = 'Covid identified'
        print('Covid identified')
    elif c_num > 0.2 * len(class_list):
        pre = 'CAP identified'
        print('CAP identified')
    else:
        pre = 'Non-infected'
        print('Non-infected')
    res.set(pre)


def selectPath():
    path_ = askdirectory()
    pathv.set(path_)


pathv = StringVar()

Label(gui, text="Result:").grid(row=0, column=0)
e1 = Entry(gui, textvariable=pathv).grid(row=0, column=1)
Button(gui, text="Select", command=selectPath).grid(row=0, column=2)

box2 = Listbox(gui)
box2.grid(row=10, column=1)
for item in ["Class:", "Covid-19", "CAP", "Non-infected"]:
    box2.insert("end", item)
res = StringVar()
res.set('No input')
result = Label(gui, textvariable=res)
result.grid(row=2, column=0)

Button(gui, text='Run Prediction', command=predict).grid(row=12, column=1)

gui.mainloop()
