from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
import cv2
from kivy.clock import Clock

class CameraApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.camera = Camera(resolution=(640, 480), play=True)
        self.texture = Texture.create(size=(640, 480), colorfmt='rgb')
        self.camera.texture = self.texture
        layout.add_widget(self.camera)

        # Panggil fungsi update setiap frame
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout

    def update(self, dt):
        # Ambil frame dari kamera
        frame = self.camera.texture.pixels

        # Konversi frame ke format yang dapat diproses oleh OpenCV
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Ubah frame menjadi grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Konversi kembali ke format yang dapat ditampilkan oleh Kivy
        gray_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Perbarui texture dengan frame grayscale
        self.texture.blit_buffer(gray_frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

if __name__ == '__main__':
    CameraApp().run()
