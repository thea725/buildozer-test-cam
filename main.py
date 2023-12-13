from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

class CameraApp(App):
    def build(self):
        # Membuat tata letak utama
        layout = BoxLayout(orientation='vertical')

        # Menambahkan objek kamera
        self.capture = cv2.VideoCapture(0)
        self.image = Image()
        layout.add_widget(self.image)

        # Mendaftarkan fungsi update untuk mengambil frame kamera secara berkala
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        # Membaca frame dari kamera
        ret, frame = self.capture.read()

        if ret:
            # Mengonversi frame menjadi grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Mengubah format frame menjadi RGB
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Membuat objek texture Kivy dari frame
            texture = Texture.create(size=(rgb_frame.shape[1], rgb_frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(rgb_frame.tostring(), colorfmt='rgb', bufferfmt='ubyte')

            # Menampilkan frame di widget gambar
            self.image.texture = texture

if __name__ == '__main__':
    CameraApp().run()
