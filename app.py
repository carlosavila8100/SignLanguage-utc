import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from A2SL.main_recorder import recognizer_speech
from A2SL.file_audio import file_audio_recognise
from A2SL.text_processing import Text_processing
from tkinter import filedialog
import os
import torch  # Importar YOLOv5

class Frames(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(0, weight=1)
        self.text_box = ""
        self.filtered_text = ""
        self.video_source = "{}/assets/".format(os.getcwd())

        self.My_Frame_1()
        self.My_Frame_2()
        self.My_Frame_3()

    def My_Frame_1(self):
        self.frame1 = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.frame1.grid(row=0, column=0, rowspan=4, padx=10, pady=10, sticky="nsew")
        self.frame1.grid_rowconfigure(3, weight=1)

        self.title_label = ctk.CTkLabel(self.frame1, text="Signals", font=ctk.CTkFont(size=25, weight="bold"))
        self.title_label.grid(row=0, column=0, rowspan=1, padx=20, pady=30)

        self.record_button = ctk.CTkButton(self.frame1, text="Grabar audio", fg_color=("#4cbb17","#209920"),
                                           hover_color=("#255215","#1C791D"), corner_radius=10,
                                           command=self.record_audio, height=30)
        self.record_button.grid(row=1, column=0, padx=20, pady=30)

        self.audio_select_button = ctk.CTkButton(self.frame1, text="Seleccionar archivo", command=self.open_file)
        self.audio_select_button.grid(row=2, column=0, padx=20, pady=30)

        self.audio_status = ctk.CTkLabel(self.frame1, text="", height=20)
        self.audio_status.grid(row=3, column=0, padx=20, pady=30)

        self.appearance_mode_label = ctk.CTkLabel(self.frame1, text="Apariencia:")
        self.appearance_mode_label.grid(row=4, column=0, padx=20, pady=(30,0))

        self.appearance_mode = ctk.CTkOptionMenu(self.frame1, values=["Clasico", "Oscuro", "Claro"],
                                                 command=self.change_appearance_mode)
        self.appearance_mode.grid(row=5, column=0, padx=20, pady=(0,30))

        # Botón "Interprete" para abrir la ventana de detección de manos
        self.interpreter_button = ctk.CTkButton(self.frame1, text="Interprete", command=self.open_hand_tracking)
        self.interpreter_button.grid(row=6, column=0, padx=20, pady=30)

    def My_Frame_2(self):
        self.frame2 = ctk.CTkFrame(self, corner_radius=2)
        self.frame2.grid(row=0, column=1, rowspan=4, padx=10, pady=10, sticky="nsew")
        self.frame2.grid_rowconfigure(1, weight=1)
        self.frame2.grid_rowconfigure(3, weight=1)

        self.recognized_text_label = ctk.CTkLabel(self.frame2, text="Text")
        self.recognized_text_label.grid(row=0, column=1, padx=70)

        self.recognized_text = ctk.CTkTextbox(self.frame2, wrap="word", corner_radius=10)
        self.recognized_text.grid(row=1, column=1, padx=10, sticky="nsew", rowspan=1)

        self.processed_text_label = ctk.CTkLabel(self.frame2, text="Texto procesado")
        self.processed_text_label.grid(row=2, column=1, padx=70)

        self.processed_text = ctk.CTkTextbox(self.frame2, wrap="word", corner_radius=10)
        self.processed_text.grid(row=3, column=1, pady=(0, 10), padx=10, sticky="nsew", rowspan=1)

        self.generate_animation = ctk.CTkButton(self.frame2, text="Generar animación", command=self.animation)
        self.generate_animation.grid(row=4, column=1, padx=70, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.frame2)
        self.progress_bar.configure(mode="indeterminate")

    def My_Frame_3(self):
        self.canvas = ctk.CTkCanvas(self)
        self.canvas.grid(row=0, column=2, sticky="nsew", padx=10, pady=20)
        self.grid_columnconfigure(2, weight=1)

    def update_video(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        ret, frame = self.video_stream.read()
        if ret:
            frame = cv2.resize(frame, (canvas_width, canvas_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=photo, anchor=ctk.NW)
            self.canvas.image = photo
            self.after(20, self.update_video)
        else:
            self.current_video_index += 1
            if self.current_video_index < len(self.url_list):
                self.video_stream.release()
                self.video_stream = cv2.VideoCapture(self.url_list[self.current_video_index])
                self.update_video()

    def animation(self):
        self.url_list = []
        self.current_video_index = 0
        for word in self.filtered_text:
            word_url = "{}{}.mp4".format(self.video_source, word)
            self.url_list.append(word_url)
        self.video_stream = cv2.VideoCapture("{}Hello.mp4".format(self.video_source))
        self.update_video()

    def record_audio(self):
        self.text_box = recognizer_speech(self.audio_status)
        self.update_textboxes(self.text_box)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
        if file_path:
            audio_url = file_path.replace(" ", "%20")
            self.text_box = file_audio_recognise(audio_url)
        self.update_textboxes(self.text_box)

    def change_appearance_mode(self, new_appearance_mode: str):
        if new_appearance_mode == 'Clasico':
            ctk.set_appearance_mode("System")
        elif new_appearance_mode == 'Oscuro':
            ctk.set_appearance_mode("Dark")
        elif new_appearance_mode == 'Claro':
            ctk.set_appearance_mode("Light")

    def update_textboxes(self, text_box):
        self.recognized_text.delete("1.0", "end")
        self.recognized_text.insert("0.0", text_box)
        self.filtered_text = Text_processing(self.text_box)
        self.processed_text.delete("1.0", "end")
        self.processed_text.insert("0.0", self.filtered_text)

    def open_hand_tracking(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            results.render()

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Audio a lengua de señas")
        self.geometry("900x500")
        self.configure(background="#36454F")
        self.minsize(width=700, height=500)
        view = Frames(self)
        view.pack(fill="both", expand=True)

app = App()
app.mainloop()
