from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.clock import Clock

import cv2
from kivy.graphics.texture import Texture

class CameraApp(App):
    def build(self):
        # Membuat tata letak utama
        layout = BoxLayout(orientation='vertical')

        # Menambahkan objek kamera
        self.camera = Camera(resolution=(640, 480), play=True)
        layout.add_widget(self.camera)

        # Menambahkan objek tampilan gambar untuk menampilkan hasil pemrosesan OpenCV
        self.image = Image()
        layout.add_widget(self.image)

        # Menjadwalkan fungsi update setiap frame kamera
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        # Mendapatkan frame dari kamera
        frame = self.camera.texture
        if frame is not None:
            # Mengubah frame ke format yang dapat diolah oleh OpenCV
            frame_data = frame.pixels
            img = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2BGR)

            # Mengonversi gambar ke grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Mengubah kembali ke format yang dapat ditampilkan oleh Kivy
            buf1 = cv2.flip(gray, 0)
            buf = buf1.tobytes()

            # Menampilkan gambar hasil pemrosesan
            texture = Texture.create(size=(gray.shape[1], gray.shape[0]), colorfmt='luminance')
            texture.blit_buffer(buf, colorfmt='luminance', bufferfmt='ubyte')
            self.image.texture = texture

if __name__ == '__main__':
    CameraApp().run()
