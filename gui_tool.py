import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog 
import sys, os, cv2, shutil, base64
import ujson as json
import numpy as np
from io import BytesIO
sys.path.append(os.getcwd())

from PIL import Image, ImageTk
from functools import partial
from detect.hole.hole_detect_old import HoleDetectModel
from tool.stand_tool import StandTool
from detect.push_back.hole_push_back_old_v1 import HolePushBackModel

global global_num, hole_dict, hole_detail_vector, hole_detail, workpiece_dict, text_item, rect_item, image_loc, label_dict, cam, cam_panel, save_frame, mouse
hole_dict, hole_detail, hole_detail_vector, label_dict, workpiece_dict, text_item, rect_item = {}, [], [], {}, {}, {}, {}
root = tk.Tk()

root.title('工件辨識系統')
root.geometry('1300x700')
root.resizable(False, False)
WIDTH, HEIGHT = 960, 540
TRANSPARENT = "iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAYAAAA8AXHiAAAACXBIWXMAAAsTAAALEwEAmpwYAAAFyWlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNy4yLWMwMDAgNzkuMWI2NWE3OSwgMjAyMi8wNi8xMy0xNzo0NjoxNCAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIiB4bWxuczpwaG90b3Nob3A9Imh0dHA6Ly9ucy5hZG9iZS5jb20vcGhvdG9zaG9wLzEuMC8iIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDIzLjUgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMi0xMS0yNlQxODo1NTowMyswODowMCIgeG1wOk1ldGFkYXRhRGF0ZT0iMjAyMi0xMS0yNlQxODo1NTowMyswODowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjItMTEtMjZUMTg6NTU6MDMrMDg6MDAiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6OTliMDkyYTYtN2ZiNy04MTRmLWEwODAtYmRkM2QyYjMwNzBlIiB4bXBNTTpEb2N1bWVudElEPSJhZG9iZTpkb2NpZDpwaG90b3Nob3A6YzI3OGI4MzMtNTk4My1hZTQ3LThhYjItY2NhODZmN2U1M2Y1IiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6NzIyMTY4OWItYWI0ZS0xZjQ5LTk2ZjQtMTA2MWY2ZmQyN2Y4IiBkYzpmb3JtYXQ9ImltYWdlL3BuZyIgcGhvdG9zaG9wOkNvbG9yTW9kZT0iMyI+IDx4bXBNTTpIaXN0b3J5PiA8cmRmOlNlcT4gPHJkZjpsaSBzdEV2dDphY3Rpb249ImNyZWF0ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6NzIyMTY4OWItYWI0ZS0xZjQ5LTk2ZjQtMTA2MWY2ZmQyN2Y4IiBzdEV2dDp3aGVuPSIyMDIyLTExLTI2VDE4OjU1OjAzKzA4OjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgMjMuNSAoV2luZG93cykiLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOjk5YjA5MmE2LTdmYjctODE0Zi1hMDgwLWJkZDNkMmIzMDcwZSIgc3RFdnQ6d2hlbj0iMjAyMi0xMS0yNlQxODo1NTowMyswODowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIDIzLjUgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8+IDwvcmRmOlNlcT4gPC94bXBNTTpIaXN0b3J5PiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pi8jU7cAAAGXSURBVHic7dLBDcAgEMCw0v13PpYgQkL2BHlkzcwHp/23A3iTsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLhLFIGIuEsUgYi4SxSBiLxAZL0gQp7bQ5DAAAAABJRU5ErkJggg=="
image_loc = ""

class CanvasButton: # 洞口圈選框建立一個invisible的button以做點選
    def __init__(self, canvas, x, y, image_path, command):
        self.canvas = canvas
        self.btn_image = image_path
        self.canvas_btn_img_obj = canvas.create_image(x, y, anchor='nw', image=self.btn_image)
        canvas.tag_bind(self.canvas_btn_img_obj, "<ButtonRelease-1>",
                        lambda event: (command()))

#--------------------------------洞口順序相關副程式--------------------------------
def yolo_detect(): # 調用yolov5辨識洞口
    global image_loc, global_num, label_dict, zoomx, zoomy

    hole_list.delete(0, 'end')
    if type(image_loc) == list:
        image_loc = np.uint8(np.asarray(image_loc))
    
    hole_detail_vector, det = hole_detect.detect(image_loc)

    zoomx, zoomy = image_loc.shape[1]/WIDTH, image_loc.shape[0]/HEIGHT
    image_loc = image_loc.tolist()
    
    if hole_detail_vector:
        messagebox.showinfo("辨識完成", "洞口已辨識完成。")

        img = ImageTk.PhotoImage(Image.fromarray(np.uint8(np.asarray(image_loc))).resize((WIDTH, HEIGHT), Image.ANTIALIAS))
        canvas.background = img  # Keep a reference in case this code is put in a function.
        bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)

        btn, num, global_num = {}, 1, 1

        for i in hole_detail_vector:
            a, b, c, d= i[0][0][0], i[0][0][1], i[0][1][0], i[0][1][1]
            hole_dict.update({"A"+str(num): [a, b, c, d, -1]})

            a, b, c, d = a/zoomx, b/zoomy, c/zoomx, d/zoomy
            canvas.create_rectangle(a,b,c,d,outline="#FF0000")
            w, h = abs(a-c), abs(b-d)

            action = partial(button_click, a, b, canvas, num)
            btnimg = ImageTk.PhotoImage(Image.open(BytesIO(base64.b64decode(TRANSPARENT))).resize((int(w), int(h)), Image.ANTIALIAS))
            btn[num] = CanvasButton(canvas, a, b,btnimg , action)
            num += 1
        
        canvas.pack()
    else:
        messagebox.showerror("辨識失敗", "洞口辨識失敗!")

def button_click(a,b,canvas,num): # 點選洞口
    global global_num, hole_dict, text_item, rect_item

    flag = False
    if not "A" in str(num):
        tmp = hole_dict["A"+str(num)]
        if tmp[4] == -1:
            tmp[4], flag = global_num, True
            hole_dict.update({"A"+str(num): tmp})
    else:
        tmp = hole_dict[str(num)]
        if tmp[4] == -1:
            tmp[4], flag = global_num, True
            hole_dict.update({str(num): tmp})

    if flag:
        text_item[global_num] = canvas.create_text(a,b,fill="white",anchor="w",text=global_num,font="16", tag=("text"+str(global_num), "text"))
        bbox = canvas.bbox(text_item[global_num])
        rect_item[global_num] = canvas.create_rectangle(bbox, fill="black", tag=("rect"+str(global_num), "rect"))
        canvas.tag_raise(text_item[global_num],rect_item[global_num])

        hole_list.insert(tk.END, global_num)
        global_num += 1
    else:
        messagebox.showwarning("警告!", "已點選此孔洞。")
    
def undo(): # 刪除選取的洞口
    global global_num, hole_dict, text_item, rect_item

    try:
        current = hole_list.get(hole_list.curselection())
    except:
        messagebox.showerror("錯誤!", "請選擇欲刪除的孔洞。")
    else:
        found = [[key, item] for key, item in hole_dict.items() if str(item[4]) == str(current)]
        hole_dict[str(found[0][0])][4] = -1

        for i in range(int(current)+1, global_num):
            found = [[key, item] for key, item in hole_dict.items() if str(item[4]) == str(i)]
            hole_dict[str(found[0][0])][4] = i-1

        global_num -= 1

        # 整個canvas和list重弄比較快且穩定
        hole_list.delete(0, 'end')
        for i in range(1, global_num):
            hole_list.insert(tk.END, i)
        canvas.delete("text", "rect")
        
        for i, j  in hole_dict.items():
            if j[4] != -1:
                a, b = j[0]/zoomx, j[1]/zoomy
                text_item[i] = canvas.create_text(a,b,fill="white",anchor="w",text=j[4],font="16", tag=("text"+str(j[4]), "text"))
                bbox = canvas.bbox(text_item[i])
                rect_item[i] = canvas.create_rectangle(bbox, fill="black", tag=("rect"+str(j[4]), "rect"))
                canvas.tag_raise(text_item[i],rect_item[i])
                global_num = max(int(global_num), int(j[4]))

def reset(): # 重設所有洞口
    global global_num, text_item, rect_item, hole_dict

    text_item, rect_item, global_num = {}, {}, 1
    canvas.delete("text", "rect")
    for k ,v in hole_dict.items():
        v[4] = -1
    hole_list.delete(0, 'end')

#------------------------------json/dict互轉換副程式------------------------------
def json2dict(i):    
    out = {i['name']: [ i['real_image'], i['shape'], i['hole'], i['hole_detail'], i['length_relationship']]}
    return out

def dict2json(input):
    for k ,v in input.items():
        out = {'name':k, 'real_image': v[0], 'shape': v[1], 'hole': v[2], 'hole_detail':v[3], 'length_relationship':v[4]}
    return out

#------------------------------圈選未出現孔洞相關副程式-----------------------------
def on_button_press(event):
    global start_x, start_y, rect

    # save mouse drag start position
    start_x, start_y= event.x, event.y
    rect = canvas.create_rectangle(event.x, event.y, 1, 1, outline="#FF0000")

def on_move_press(event):
    curX, curY = (event.x, event.y)

    # expand rectangle as you drag the mouse
    canvas.coords(rect, start_x, start_y, curX, curY)

def on_button_release(event):
    global save_temp
    end_x, end_y = event.x, event.y
    save_temp.append([start_x, start_y, end_x, end_y])
    pass


def save():
    global save_temp
    for start_x, start_y, end_x, end_y in save_temp:
        w, h= abs(start_x-end_x), abs(start_y-end_y)

        action = partial(button_click, start_x,start_y,canvas, len(hole_dict))
        btnimg = ImageTk.PhotoImage(Image.open(BytesIO(base64.b64decode(TRANSPARENT))).resize((int(w), int(h)), Image.ANTIALIAS))
        btn[len(hole_dict)+1] = CanvasButton(canvas, start_x,start_y,btnimg , action)

        a, b, c, d = start_x*zoomx, start_y*zoomy, end_x*zoomx, end_y*zoomy
        hole_dict.update({"A"+str(len(hole_dict)+1):[a,b,c,d,-1]})
    
    save_temp = {}
    canvas.unbind("<ButtonPress-1>")
    canvas.unbind("<B1-Motion>")
    canvas.unbind("<ButtonRelease-1>")

def add():
    global save_temp
    save_temp = []
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)

#-------------------------------儲存工件資料相關副程式------------------------------
def save_data(name):
    global workpiece_dict, hole_dict, hole_detail_vector, hole_detail

    if name == "":
        messagebox.showerror("錯誤!", "尚未輸入工件名稱。")
    else:
        img_size = np.asarray(image_loc).shape

        for k, v in hole_dict.items():
            x, y, key = (float(v[0])+float(v[2]))/2, (float(v[1])+float(v[3]))/2, "?"
            if v[4] != -1:
                key = str(v[4])
            hole_detail.append([str(k), [(int(v[0]), int(v[1])), (int(v[2]), int(v[3]))], (int(x),int(y)), str(0)])
            
        length_list = []
        for tag1, _, middle1, _ in hole_detail :
            for tag2, _, middle2, _ in hole_detail :
                if tag1 == tag2 and middle1 == middle2 : continue
                length = [tag1, tag2, (middle2[0] - middle1[0], middle2[1] - middle1[1])]
                length_list.append(length)

        point_list = [x[0] for x in hole_detail]
        length_compare_list = []

        for point in point_list:
            start_with_length = [x for x in length_list if x[0] == point]
            for length1 in start_with_length:
                for length2 in start_with_length:
                    if length1 == length2 : continue
                    length_compare = [length1[0], [length1[1], length2[1]], [length1[2], length2[2]], _cal_angle(length1[2], length2[2]), _cal_ratio(length1[2], length2[2]) ]
                    length_compare_list.append(length_compare)


        tmp = {name: [image_loc, img_size, hole_dict, hole_detail, length_compare_list]}
        workpiece_dict.update(tmp)

        with open('stand_workpiece\\' + name + '.json', "w") as f:
            json.dump(dict2json(tmp), f)

        messagebox.showinfo("儲存成功!", "工件 " + name + " 已成功儲存。")
        hole_detail_vector, hole_detail = [], []
       
        if name in workpiece_list.get(0, "end"):
            pass
        else: 
            workpiece_list.insert(tk.END, name)

def _cal_angle(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / norm))
    theta = np.rad2deg(np.arccos(np.dot(v1, v2) / norm))

    if rho < 0:
        return - theta
    else:
        return theta

def _cal_ratio(v1, v2):
    length1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    length2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

    return length2 / length1

def open_img():
    global image_loc, hole_dict
    
    hole_dict = {}
    reset()
    save_name.delete(0, tk.END)
    image_loc = cv2.cvtColor(cv2.resize(cv2.imread(filedialog.askopenfilename()), (1440,810)), cv2.COLOR_BGR2RGB)

    img = ImageTk.PhotoImage(Image.fromarray(image_loc).resize((WIDTH, HEIGHT), Image.ANTIALIAS))
    canvas.background = img  # Keep a reference in case this code is put in a function.
    bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)


#-------------------------------資料庫存取相關副程式-------------------------------
def delete():
    global rect_item, text_item, workpiece_dict

    msg_box= ""
    try:
        current = workpiece_list.get(workpiece_list.curselection())
        msg_box = tk.messagebox.askquestion('刪除工件', '確定要刪除工件嗎?', icon='warning')
        
    except:
        messagebox.showerror("錯誤!", "請從資料庫選擇工件。")

    if msg_box == 'yes':
        workpiece_dict.pop(current)
        canvas.delete("all")
        workpiece_list.delete(workpiece_list.get(0, tk.END).index(current))
        hole_list.delete(0, 'end')
        os.remove('stand_workpiece\\' + current + '.json' )

        save_name.delete(0, tk.END)

def show():
    global rect_item, text_item, global_num, hole_dict, image_loc, label_dict

    try:
        current = workpiece_list.get(workpiece_list.curselection())

        text_item, rect_item, btn, global_num = {}, {}, {}, 1
        hole_list.delete(0, 'end')
        canvas.delete("all")

        save_name.delete(0, tk.END)
        save_name.insert(0, current)
        current = workpiece_dict[current]
        image_loc, hole_dict = current[0], current[2]

        img = ImageTk.PhotoImage(Image.fromarray(np.uint8(np.asarray(image_loc))).resize((WIDTH, HEIGHT), Image.ANTIALIAS))
        canvas.background = img  # Keep a reference in case this code is put in a function.
        bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)
        
        global zoomx, zoomy
        zoomx, zoomy = float(current[1][1])/WIDTH, float(current[1][0])/HEIGHT

        flag = False
        for l1, l2 in hole_dict.items():
            a,b,c,d = float(l2[0])/zoomx, float(l2[1])/zoomy, float(l2[2])/zoomx, float(l2[3])/zoomy
            w,h = abs(a-c), abs(b-d)

            action = partial(button_click, a,b,canvas, l1)
            btnimg = ImageTk.PhotoImage(Image.open(BytesIO(base64.b64decode(TRANSPARENT))).resize((int(w), int(h)), Image.ANTIALIAS))
            btn[l1] = CanvasButton(canvas, a, b,btnimg , action)

            canvas.create_rectangle(a,b,c,d,outline="#FF0000")
            if l2[4] != -1:
                flag = True
                text_item[l1] = canvas.create_text(a,b,fill="white",anchor="w",text=l2[4],font="16", tag=("text"+str(l2[4]), "text"))
                bbox = canvas.bbox(text_item[l1])
                rect_item[l1] = canvas.create_rectangle(bbox, fill="black", tag=("rect"+str(l2[4]), "rect"))
                canvas.tag_raise(text_item[l1],rect_item[l1])
                global_num = max(int(global_num), int(l2[4]))

        if flag:
            for i in range(1, global_num+1):
                hole_list.insert(tk.END, i)
            
            global_num += 1
        canvas.pack()
    except :
        messagebox.showerror("錯誤!", "請從資料庫選擇工件。")
    
#----------------------------------相機相關副程式----------------------------------
def start_cam():
    global frame, cam, cam_panel

    close_btn.configure(state="active")
    snap_btn.configure(state="active")
    cam_btn.configure(state="disabled")

    cam_panel=tk.Label(root) 
    cam_panel.place(x=170, y=0)

    reset()
    
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 810)

    while True:
        ret, frame = cam.read()

        #Update the image to tkinter...
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_update = ImageTk.PhotoImage(Image.fromarray(frame).resize((WIDTH, HEIGHT), Image.ANTIALIAS))
        cam_panel.configure(image=img_update, width=WIDTH, height=HEIGHT)
        cam_panel.image=img_update
        cam_panel.update()

        if not ret:
            print("failed to grab frame")
            break
        cv2.waitKey(0)

def capture_cam():
    global cam, frame, save_frame
    save_frame = frame
    messagebox.showinfo("成功", "已拍攝照片。")

def close_cam():
    global cam, save_frame, cam_panel, image_loc, zoomx, zoomy, hole_dict
    
    close_btn.configure(state="disabled")
    snap_btn.configure(state="disabled")
    cam_btn.configure(state="active")

    try:
        image_loc = cv2.resize(save_frame, (1440,810))
        zoomx, zoomy = image_loc.shape[1]/WIDTH, image_loc.shape[0]/HEIGHT

        hole_dict = {}
        reset()
        img = ImageTk.PhotoImage(Image.fromarray(image_loc).resize((WIDTH, HEIGHT), Image.ANTIALIAS))
        canvas.background = img  # Keep a reference in case this code is put in a function.
        bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)
    except NameError:
        pass 
    cam_panel.destroy()
    try:
        cam.release()
        cv2.destroyAllWindows()
    except:
        pass
    
    messagebox.showinfo("關閉相機", "已關閉相機。")

#--------------------------------即時辨識相關副程式--------------------------------
def realtime():
    current = ""
    try:
        current = workpiece_list.get(workpiece_list.curselection())
    except:
        messagebox.showwarning("警告!", "請先從工件列表選取欲辨識的工件。")

    if current:
        detect(current)

def detect(name):
    temp = {"parent_foleder": config['stand']["parent_foleder"], "stand_tool_name":name}
    push_back = HolePushBackModel(config['push_back'], StandTool(temp))
 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    cv2.namedWindow("Result", 0)
    cv2.resizeWindow("Result", 1280, 720)
    
    while True :
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        try:
            hole_detail_vector, det  = hole_detect.detect(frame)
            push_back_list, min_hit = push_back.get_push_back_position((frame.shape[0], frame.shape[1]), hole_detail_vector)
            push_back_frame = push_back.get_hole_frame(frame, push_back_list, min_hit)
            cv2.imshow("Result", push_back_frame)

        except:
            cv2.imshow("Result", frame)
            pass

        if cv2.waitKey(1) == ord('q'):
            break
        
        if cv2.getWindowProperty("Result",cv2.WND_PROP_VISIBLE) < 1:        
            break


# Load config
with open('config.json') as f:
    config = json.load(f)
hole_detect = HoleDetectModel(config['yolov5'])

# Initialize the variables
btn, text_item, rect_item, num, global_num = {}, {}, {}, 1, 1

# Initialize the canvas
canvas = tk.Canvas(root, width = WIDTH, height= HEIGHT)

# Initialize and place the widgets
workpiece_label = tk.Label(root, text='工件列表', font=('微軟正黑體', 16))
workpiece_label.place(x=10,y=65)

workpiece_list = tk.Listbox(root, height=15)
workpiece_list.place(x=10,y=100)

hole_label = tk.Label(root, text='已點選洞口', font=('微軟正黑體', 16))
hole_label.place(x=1145,y=65)

hole_list = tk.Listbox(root, height=15)
hole_list.place(x=1145,y=100)

show_btn = tk.Button(root, text = '顯示選取的工件', command=show, height=2, width=15)
show_btn.place(x=24,y=350)

delete_btn = tk.Button(root, text ='刪除選取的工件', command=delete, height=2, width=15)
delete_btn.place(x=24,y=400)

save_name = tk.Entry(root, width=15)
save_name.place(x=30,y=450)

save_btn = tk.Button(root, text = '儲存工件到資料庫', command=lambda: save_data(save_name.get()), height=2, width=15)
save_btn.place(x=24, y=480)

undo = tk.Button(root, text = '刪除選取的洞口', command=undo, height=2, width=15)
undo.place(x=1160,y=350)

reset_btn = tk.Button(root, text = '重設所有洞口', command=reset, height=2, width=15)
reset_btn.place(x=1160,y=400)

add_btn = tk.Button(root, text = '圈選未出現洞口', command=add, height=2, width=15)
add_btn.place(x=1160,y=450)

save_add_btn = tk.Button(root, text = '儲存圈選的洞口', command=save, height=2, width=15)
save_add_btn.place(x=1160,y=500)

imgbtn = tk.Button(root, text ='從本機載入圖片', command=open_img, height=2, width=15)
imgbtn.place(x=200,y=570)

detect_btn = tk.Button(root, text = '開始辨識洞口', command=yolo_detect, height=2, width=15)
detect_btn.place(x=365,y=570)

cam_btn = tk.Button(root, text = '開啟相機', command=start_cam, height=2, width=15)
cam_btn.place(x=515,y=570)

snap_btn = tk.Button(root, text = '拍攝畫面', command=capture_cam, state="disabled", height=2, width=15)
snap_btn.place(x=665,y=570)

close_btn = tk.Button(root, text = '關閉相機', command=close_cam, state="disabled", height=2, width=15)
close_btn.place(x=815,y=570)

realtime_btn = tk.Button(root, text = '即時辨識', command=realtime, height=2, width=15)
realtime_btn.place(x=975,y=570)


toollist = os.listdir('stand_workpiece')
for i in toollist:
    with open('stand_workpiece\\' + i) as f:
        i=i.replace(".json", "")
        workpiece_dict.update(json2dict(json.load(f)))
        workpiece_list.insert(tk.END, i)

canvas.pack()
root.mainloop()