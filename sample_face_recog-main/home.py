import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import threading
import shutil
from facerec import *
from register import *
from face_detection import *
from dbHandler import *
#from handler import *
import time
import csv
import numpy as np
import ntpath
import os
#from live_cam import *

active_page = 0
thread_event = None
left_frame = None
right_frame = None
heading = None
webcam = None
img_label = None
img_read = None
img_list = []
slide_caption = None
slide_control_panel = None
current_slide = -1

root = tk.Tk()
root.geometry("1000x900+200+100")

# create Pages
pages = []
for i in range(5):
    pages.append(tk.Frame(root, bg="#3E3B3C"))
    pages[i].pack(side="top", fill="both", expand=True)
    pages[i].place(x=0, y=0, relwidth=1, relheight=1)


def goBack():
    global active_page, thread_event, webcam

    if (active_page==4 and not thread_event.is_set()):
        thread_event.set()
        webcam.release()

    for widget in pages[active_page].winfo_children():
        widget.destroy()

    pages[0].lift()
    active_page = 0


def basicPageSetup(pageNo):
    global left_frame, right_frame, heading

    back_img = tk.PhotoImage(file= r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\img\back.png")
    back_button = tk.Button(pages[pageNo], image=back_img, bg="#3E3B3C", bd=0, highlightthickness=0,
           activebackground="#3E3B3C", command=goBack)
    back_button.image = back_img
    back_button.place(x=10, y=10)

    heading = tk.Label(pages[pageNo], fg="white", bg="#3E3B3C", font="Arial 20 bold", pady=10)
    heading.pack()

    content = tk.Frame(pages[pageNo], bg="#3E3B3C", pady=20)
    content.pack(expand="true", fill="both")

    left_frame = tk.Frame(content, bg="#3E3B3C")
    left_frame.grid(row=0, column=0, sticky="nsew")

    right_frame = tk.LabelFrame(content, text="Detected Criminals", bg="#3E3B3C", font="Arial 20 bold", bd=4,
                             foreground="#2ea3ef", labelanchor="n")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    content.grid_columnconfigure(0, weight=1, uniform="group1")
    content.grid_columnconfigure(1, weight=1, uniform="group1")
    content.grid_rowconfigure(0, weight=1)


def showImage(frame, img_size):
    global img_label, left_frame

    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    if (img_label == None):
        img_label = tk.Label(left_frame, image=img, bg="#202d42")
        img_label.image = img
        img_label.pack(padx=20)
    else:
        img_label.configure(image=img)
        img_label.image = img


def getNewSlide(control):
    global img_list, current_slide

    if(len(img_list) > 1):
        if(control == "prev"):
            current_slide = (current_slide-1) % len(img_list)
        else:
            current_slide = (current_slide+1) % len(img_list)

        img_size = left_frame.winfo_height() - 200
        showImage(img_list[current_slide], img_size)

        slide_caption.configure(text = "Image {} of {}".format(current_slide+1, len(img_list)))


def selectMultiImage(opt_menu, menu_var):
    global img_list, current_slide, slide_caption, slide_control_panel

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path_list = filedialog.askopenfilenames(title="Choose atleast 5 images", filetypes=filetype)

    if(len(path_list) < 5):
        messagebox.showerror("Error", "Choose atleast 5 images.")
    else:
        img_list = []
        current_slide = -1

        # Resetting slide control panel
        if (slide_control_panel != None):
            slide_control_panel.destroy()

        # Creating Image list
        for path in path_list:
            img_list.append(cv2.imread(path))

        # Creating choices for profile pic menu
        menu_var.set("")
        opt_menu['menu'].delete(0, 'end')

        for i in range(len(img_list)):
            ch = "Image " + str(i+1)
            opt_menu['menu'].add_command(label=ch, command= tk._setit(menu_var, ch))
            menu_var.set("Image 1")


        # Creating slideshow of images
        img_size =  left_frame.winfo_height() - 200
        current_slide += 1
        showImage(img_list[current_slide], img_size)

        slide_control_panel = tk.Frame(left_frame, bg="#202d42", pady=20)
        slide_control_panel.pack()

        back_img = tk.PhotoImage(file=r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\img\previous.png")
        next_img = tk.PhotoImage(file=r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\img\next.png")

        prev_slide = tk.Button(slide_control_panel, image=back_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("prev"))
        prev_slide.image = back_img
        prev_slide.grid(row=0, column=0, padx=60)

        slide_caption = tk.Label(slide_control_panel, text="Image 1 of {}".format(len(img_list)), fg="#ff9800",
                              bg="#202d42", font="Arial 15 bold")
        slide_caption.grid(row=0, column=1)

        next_slide = tk.Button(slide_control_panel, image=next_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("next"))
        next_slide.image = next_img
        next_slide.grid(row=0, column=2, padx=60)



def register(entries, required, menu_var):
    global img_list

    # Checking if no image is selected
    if len(img_list) == 0:
        messagebox.showerror("Error", "Select Images first.")
        return

    # Fetching data from entries
    entry_data = {}
    for i, entry in enumerate(entries):
        val = entry[1].get()

        if len(val) == 0 and required[i] == 1:
            messagebox.showerror("Field Error", f"Required field missing :\n\n{entry[0]}")
            return
        else:
            entry_data[entry[0]] = val.lower()

    # Creating a directory for the criminal based on their name
    criminal_name = entry_data["Name"].replace(" ", "_")  # Replace spaces with underscores for the folder name
    criminal_dir = os.path.join(r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\face_samples", criminal_name)
    if not os.path.isdir(criminal_dir):
        os.mkdir(criminal_dir)

    no_face = []
    for i, img in enumerate(img_list):
        # Storing Images in the directory
        id = registerCriminal(img, criminal_dir, i + 1)  # Save images to the criminal's folder
        if id is not None:
            no_face.append(id)

    # Check if any image doesn't contain a face
    if len(no_face) > 0:
        no_face_st = ", ".join(f"Image {i}" for i in no_face)
        messagebox.showerror("Registration Error", f"Registration failed!\n\nFollowing images don't contain "
                                                    f"face or the face is too small:\n\n{no_face_st}")
        shutil.rmtree(criminal_dir, ignore_errors=True)  # Clean up the created folder
    else:
        # Storing data in the CSV file
        with open("criminals.csv", mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Writing the details to the CSV
            writer.writerow([
                entry_data["Name"],
                entry_data.get("Father_Name", ""),
                entry_data.get("Gender", ""),
                entry_data.get("DOB(yyyy-mm-dd)", ""),
                entry_data.get("Crimes_Done", "")
            ])
        
        

        
        # Call insertData function to store data in the database
        rowId = insertData(entry_data)  # This will store the data in MySQL and return the rowId

        if rowId >= 0:
            messagebox.showinfo("Success", "Criminal Registered Successfully.")

            # Save profile pic using the criminal's name
            profile_img_num = int(menu_var.get().split(' ')[1]) - 1
            profile_pics_dir = r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\profile_pics"
            if not os.path.isdir(profile_pics_dir):
                os.mkdir(profile_pics_dir)

            # Use the criminal's name for the profile picture filename
            profile_pic_filename = f"{criminal_name}_profile.png"  # or any other format you want
            cv2.imwrite(os.path.join(profile_pics_dir, profile_pic_filename), img_list[profile_img_num])

            goBack()  # Assuming this function handles the UI navigation
        else:
            shutil.rmtree(criminal_dir, ignore_errors=True)
            messagebox.showerror("Database Error", "Some error occurred while storing data.")

## update scrollregion when all widgets are in canvas
def on_configure(event, canvas, win):
    canvas.configure(scrollregion=canvas.bbox('all'))
    canvas.itemconfig(win, width=event.width)

## Register Page ##
def getPage1():
    global active_page, left_frame, right_frame, heading, img_label
    active_page = 1
    img_label = None
    opt_menu = None
    menu_var = tk.StringVar(root)
    pages[1].lift()

    basicPageSetup(1)
    heading.configure(text="Register Criminal", bg="#3E3B3C")
    right_frame.configure(text="Enter Details", fg="white", bg="#3E3B3C")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Images", command=lambda: selectMultiImage(opt_menu, menu_var), font="Arial 15 bold", bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").grid(row=0, column=0, padx=25, pady=25)


    # Creating Scrollable Frame
    canvas = tk.Canvas(right_frame, bg="#202d42", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand="true", padx=30)
    scrollbar = tk.Scrollbar(right_frame, command=canvas.yview, width=20, troughcolor="#3E3B3C", bd=0,
                          activebackground="#3E3B3C", bg="#000000", relief="raised")
    scrollbar.pack(side="left", fill="y")

    scroll_frame = tk.Frame(canvas, bg="#3E3B3C", pady=20)
    scroll_win = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda event, canvas=canvas, win=scroll_win: on_configure(event, canvas, win))


    tk.Label(scroll_frame, text="* Required Fields", bg="#3E3B3C", fg="yellow", font="Arial 13 bold").pack()
    # Adding Input Fields
    input_fields = ("Name", "Father_Name", "Gender", "DOB(yyyy-mm-dd)", "Crimes_Done", "Profile Image")
    ip_len = len(input_fields)
    required = [1, 1, 1, 1, 1, 1]

    entries = []
    for i, field in enumerate(input_fields):
        print()
        row = tk.Frame(scroll_frame, bg="#3E3B3C")
        row.pack(side="top", fill="x", pady=15)

        label = tk.Text(row, width=20, height=1, bg="#3E3B3C", fg="#ffffff", font="Arial 13", highlightthickness=0, bd=0)
        label.insert("insert", field)
        label.pack(side="left")

        if(required[i] == 1):
            label.tag_configure("star", foreground="yellow", font="Arial 13 bold")
            label.insert("end", "  *", "star")
        label.configure(state="disabled")

        if(i != ip_len-1):
            ent = tk.Entry(row, font="Arial 13", selectbackground="#90ceff")
            ent.pack(side="right", expand="true", fill="x", padx=10)
            entries.append((field, ent))
        else:
            menu_var.set("Image 1")
            choices = ["Image 1"]
            opt_menu = tk.OptionMenu(row, menu_var, *choices)
            opt_menu.pack(side="right", fill="x", expand="true", padx=10)
            opt_menu.configure(font="Arial 13", bg="#000000", fg="white", bd=0, highlightthickness=0, activebackground="#3E3B3C")
            menu = opt_menu.nametowidget(opt_menu.menuname)
            menu.configure(font="Arial 13", bg="white", activebackground="#90ceff", bd=0)

    # print(entries)

    tk.Button(scroll_frame, text="Register", command=lambda: register(entries, required, menu_var), font="Arial 15 bold",
           bg="#000000", fg="white", pady=10, padx=30, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").pack(pady=25)

import datetime;
def showCriminalProfile(name):
    top = tk.Toplevel(bg="#202d42")
    top.title("Criminal Profile")
    top.geometry("1500x900+%d+%d" % (root.winfo_x() + 10, root.winfo_y() + 10))

    tk.Label(top, text="Criminal Profile", fg="white", bg="#202d42", font="Arial 20 bold", pady=10).pack()

    content = tk.Frame(top, bg="#202d42", pady=20)
    content.pack(expand="true", fill="both")
    content.grid_columnconfigure(0, weight=3, uniform="group1")
    content.grid_columnconfigure(1, weight=5, uniform="group1")
    content.grid_rowconfigure(0, weight=1)

    # Replace underscores with spaces in the name for retrieving data
    name_with_spaces = name.replace("_", " ")
    (id, crim_data) = retrieveData(name_with_spaces)

    # Use the name with underscores for the image file
    name_with_underscores = name.replace(" ", "_")
    path = os.path.join(r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\profile_pics", f"{name_with_underscores}_profile.png")
    print(f"Loading image from: {path}")
    
    profile_img = cv2.imread(path)
    if profile_img is None:
        print(f"Image not found for {name_with_underscores}")
        return

    # Resize and display the image
    profile_img = cv2.resize(profile_img, (500, 500))
    img = cv2.cvtColor(profile_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(content, image=img, bg="#202d42")
    img_label.image = img  # Keep a reference to the image to prevent garbage collection
    img_label.grid(row=0, column=0)

    info_frame = tk.Frame(content, bg="#202d42")
    info_frame.grid(row=0, column=1, sticky='w')

    for i, item in enumerate(crim_data.items()):
        tk.Label(info_frame, text=item[0], pady=15, fg="yellow", font="Arial 15 bold", bg="#202d42").grid(row=i, column=0, sticky='w')
        tk.Label(info_frame, text=":", fg="yellow", padx=50, font="Arial 15 bold", bg="#202d42").grid(row=i, column=1)
        
        val = item[1]

        # Check if the value is a date object and format it
        if isinstance(val, datetime.date):
            val = val.strftime("%Y-%m-%d")

        # Apply capitalize only to strings
        elif isinstance(val, str):
            val = val.capitalize()

        # Display the value
        tk.Label(info_frame, text=val, fg="white", font="Arial 15", bg="#202d42").grid(row=i, column=2, sticky='w')

def startRecognition():
    global img_read, img_label

    if(img_label == None):
        messagebox.showerror("Error", "No image selected. ")
        return

    crims_found_labels = []
    for wid in right_frame.winfo_children():
        wid.destroy()

    frame = cv2.flip(img_read, 1, 0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coords = detect_faces(gray_frame)

    if (len(face_coords) == 0):
        messagebox.showerror("Error", "Image doesn't contain any face or face is too small.")
    else:
        (model, names) = train_model()
        print('Training Successful. Detecting Faces')
        (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

        img_size = left_frame.winfo_height() - 40
        frame = cv2.flip(frame, 1, 0)
        showImage(frame, img_size)

        if (len(recognized) == 0):
            messagebox.showerror("Error", "No criminal recognized.")
            return

        for i, crim in enumerate(recognized):
            crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg="orange",
                                            font="Arial 15 bold", pady=20))
            crims_found_labels[i].pack(fill="x", padx=20, pady=10)
            crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]:showCriminalProfile(name))


def selectImage():
    global left_frame, img_label, img_read
    for wid in right_frame.winfo_children():
        wid.destroy()

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path = filedialog.askopenfilename(title="Choose a image", filetypes=filetype)

    if(len(path) > 0):
        img_read = cv2.imread(path)

        img_size =  left_frame.winfo_height() - 40
        showImage(img_read, img_size)


## Detection Page ##
def getPage2():
    global active_page, left_frame, right_frame, img_label, heading
    img_label = None
    active_page = 2
    pages[2].lift()

    basicPageSetup(2)
    heading.configure(text="Detect Criminal")
    right_frame.configure(text="Detected Criminals", fg="white")

    btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Image", command=selectImage, font="Arial 15 bold", padx=20, bg="#000000",
            fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
            activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    tk.Button(btn_grid, text="Recognize", command=startRecognition, font="Arial 15 bold", padx=20, bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
           activeforeground="white").grid(row=0, column=1, padx=25, pady=25)

# def path_leaf(path):
#     head,tail = ntpath.split(path)


def videoLoop(path,model, names):
    p=path
    q=ntpath.basename(p)
    filenam, file_extension = os.path.splitext(q)
    # print(filename)
    global thread_event, left_frame, webcam, img_label
    start=time.time()
    webcam = cv2.VideoCapture(p)
    old_recognized = []
    crims_found_labels = []
    times = []
    img_label = None
    field=['S.No.', 'Name', 'Time']
    g=filenam+'.csv'
    # filename = "g.csv"
    filename = g
    # with open('people.csv', 'w', ) as csvfile:
    # peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # os.path.join(path, vid.split('.')[0]+'_'+str(count)+'.png'
    num=0
    try:
        # with open('people_Details.csv', 'w', ) as csvfile:
        with open(filename, 'w') as csvfile:
            # peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field)   
            while not thread_event.is_set():
                
                # Loop until the camera is working
                
                    
                    while (True):
                        # Put the image from the webcam into 'frame'
                        (return_val, frame) = webcam.read()
                        if (return_val == True):
                            break
                        # else:
                        #     print("Failed to open webcam. Trying again...")

                    # Flip the image (optional)
                    frame = cv2.flip(frame, 1, 0)
                    # Convert frame to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect Faces
                    face_coords = detect_faces(gray_frame)
                    (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

                    # Recognize Faces
                    recog_names = [item[0] for item in recognized]
                    if(recog_names != old_recognized):
                        for wid in right_frame.winfo_children():
                            wid.destroy()
                        del(crims_found_labels[:])

                        for i, crim in enumerate(recognized):
                            num+=1
                            x=time.time()-start
                            crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg="orange",
                                                            font="Arial 15 bold", pady=20))
                            crims_found_labels[i].pack(fill="x", padx=20, pady=10)
                            crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]: showCriminalProfile(name))
                            y=crim[0]
                            print(x,y)
                            arr = [num,y,x]
                            # peoplewriter.writerow(arr)
                            csvwriter.writerow(arr)  
                            
                            # print('hello')
                        old_recognized = recog_names

                    # Display Video stream
                    img_size = min(left_frame.winfo_width(), left_frame.winfo_height()) - 20

                    showImage(frame, img_size)

    except RuntimeError:
        print("[INFO]Caught Runtime Error")
    except tk.TclError:
        print("[INFO]Caught Tcl Error")


# video surveillance Page ##
def getPage4(path):
    p=path
    # print(p)
    global active_page, video_loop, left_frame, right_frame, thread_event, heading
    active_page = 4
    pages[4].lift()

    basicPageSetup(4)
    heading.configure(text="Video Surveillance")
    right_frame.configure(text="Detected Criminals")
    left_frame.configure(pady=40)

    btn_grid = tk.Frame(right_frame, bg="#3E3B3C")
    btn_grid.pack()

    (model, names) = train_model()
    print('Training Successful. Detecting Faces')

    thread_event = threading.Event()
    thread = threading.Thread(target=videoLoop, args=(p,model, names))
    thread.start()

def getPage3():
    global active_page, video_loop, left_frame, right_frame, thread_event, heading
    active_page = 3
    pages[3].lift()

    basicPageSetup(3)
    heading.configure(text="Video Surveillance")

    btn_grid = tk.Frame(left_frame,bg="#3E3B3C")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Video", command=selectvideo, font="Arial 15 bold", padx=20, bg="#000000",
                fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
                activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    
    

    # tk.Button(btn_grid, text="Recognize", command=getPage3(), font="Arial 15 bold", padx=20, bg="#000000",
    #        fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
    #        activeforeground="white").grid(row=0, column=1, padx=25, pady=25)


def selectvideo():
    global left_frame, img_label, img_read
    for wid in right_frame.winfo_children():
        wid.destroy()

    filetype = [("video", "*.mp4 *.mkv")]
    path = filedialog.askopenfilename(title="Choose a video", filetypes=filetype)
    p=''
    p=path
    print(p)
    if(len(path) > 0):
        # vid_read = cv2.imread(path)
        # print(vid_read)
        getPage4(p)
        # img_read = cv2.imread(path)

    #     img_size =  left_frame.winfo_height() - 40
    #     showImage(img_read, img_size)

# def getPage3():
#     global active_page, left_frame, right_frame, img_label, heading
#     img_label = None
#     active_page = 2
#     pages[2].lift()

#     basicPageSetup(2)
#     heading.configure(text="Video Surveillance")
#     right_frame.configure(text="Detected Criminals")

#     btn_grid = tk.Frame(left_frame, bg="#3E3B3C")
#     btn_grid.pack()

#     tk.Button(btn_grid, text="Select video", command=selectvideo, font="Arial 15 bold", padx=20, bg="#000000",
#             fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
#             activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
#     tk.Button(btn_grid, text="Recognize", command=startRecognition, font="Arial 15 bold", padx=20, bg="#000000",
#            fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#3E3B3C",
#            activeforeground="white").grid(row=0, column=1, padx=25, pady=25)
#def live_cam():
   # live_cam1()

def selectvideo1():
    # global left_frame, img_label, img_read
    # for wid in right_frame.winfo_children():
    #     wid.destroy()

    filetype = [("video", "*.mp4 *.mkv")]
    path = filedialog.askopenfilename(title="Choose a video", filetypes=filetype)
    p=''
    p=path
    
    if(len(path) > 0):
        # vid_read = cv2.imread(path)
        # print(vid_read)
       detect(p)

######################################## Home Page ####################################
tk.Label(pages[0], text="Face Recognition System for Criminal Detection", fg="black", bg="#3E3B3C",
      font="Arial 25 bold", pady=30).pack()

logo = tk.PhotoImage(file = r"D:\MACHINE LEARNING\Facial-Recognition-for-Crime-Detection-master\Facial-Recognition-for-Crime-Detection-master\img\logo2.png")
tk.Label(pages[0], image=logo, bg="#3E3B3C").pack(side='left')

btn_frame = tk.Frame(pages[0], bg="#3E3B3C", pady=0)
btn_frame.pack()
tk.Button(btn_frame, text="Input Video", command=selectvideo1)
tk.Button(btn_frame, text="Add Criminal Details", command=getPage1)
#tk.Button(btn_frame, text="live Cam classification", command =live_cam)
tk.Button(btn_frame, text="Image Surveillance", command=getPage2)
tk.Button(btn_frame, text="Video Surveillance", command=getPage3)


for btn in btn_frame.winfo_children():
    btn.configure(font="Arial 20", width=17, bg="#000000", fg="white",
        pady=15, bd=0, highlightthickness=0, activebackground="#3E3B3C", activeforeground="white")
    btn.pack(pady=30)


pages[0].lift()
root.mainloop()