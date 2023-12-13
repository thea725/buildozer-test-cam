# Import library Kivy dan Plyer
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from plyer import camera

# Import library OpenCV
import cv2
from PIL import Image as PILImage
from io import BytesIO
import numpy as np

class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)

        # Fungsi untuk mengambil dan memproses frame dari kamera
        def update_frame(dt):
            frame = camera.take_picture()
            if frame:
                # Mengonversi frame ke format array NumPy
                frame_np = np.array(PILImage.open(BytesIO(frame)))

                # Mengubah gambar menjadi grayscale
                gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)

                # Menampilkan gambar di antarmuka Kivy
                self.image.texture = self.convert_frame_to_texture(gray_frame)

        # Mengatur interval untuk pembaruan frame (misalnya, setiap 1/30 detik)
        Clock.schedule_interval(update_frame, 1/30)

        return self.layout

    def convert_frame_to_texture(self, frame):
        # Mengonversi frame OpenCV menjadi format yang dapat ditampilkan oleh Kivy
        image_texture = PILImage.fromarray(frame).tostring()
        return image_texture

if __name__ == '__main__':
    CameraApp().run()
