from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.image import Image

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class CameraApp(App):
    def build(self):
        # Membuat tata letak utama
        layout = BoxLayout(orientation='vertical')

        # Menambahkan objek kamera
        self.camera = Camera(resolution=(640, 480), play=True)
        layout.add_widget(self.camera)

        # Menambahkan objek gambar untuk menampilkan hasil olahan gambar
        self.image = Image()
        layout.add_widget(self.image)

        # Menjadwalkan pemanggilan fungsi update_frame setiap frame
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        return layout

    def update_frame(self, dt):
        # Mengambil frame dari kamera
        frame = self.camera.texture

        # Mengonversi frame ke skala abu-abu menggunakan OpenCV
        if frame is not None:
            frame = self.convert_to_grayscale(frame)

            # Menyusun frame hasil olahan untuk ditampilkan di objek gambar
            self.image.texture = frame

    def convert_to_grayscale(self, frame):
        # Mengambil data dari texture
        buf = frame.pixels

        # Mengonversi data ke array NumPy
        img = cv2.flip(
            cv2.cvtColor(
                np.frombuffer(buf, dtype=np.uint8).reshape((frame.height, frame.width, 4)),
                cv2.COLOR_RGBA2BGR
            ),
            0
        )

        # Mengonversi frame ke skala abu-abu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mengonversi kembali ke format RGBA
        buf[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGBA).flatten()

        return frame

if __name__ == '__main__':
    CameraApp().run()
