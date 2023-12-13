from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import numpy as np

class AndroidCamera(BoxLayout):
    def __init__(self, **kwargs):
        super(AndroidCamera, self).__init__(**kwargs)

        self.camera = Camera(play=True, index=0)
        self.camera.bind(on_tex=self.update_texture)

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)

        return self.layout

    def update_texture(self, instance, texture):
        w, h = self.camera.resolution
        frame = np.frombuffer(self.camera._camera._buffer.tostring(), dtype='uint8').reshape((h + h // 2, w))
        frame_bgr = np.rot90(cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21), 3)
        
        # Buat tekstur Kivy dari citra OpenCV
        texture = Texture.create(size=(frame_bgr.shape[1], frame_bgr.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_bgr.tostring(), colorfmt='rgb', bufferfmt='ubyte')

        # Tampilkan gambar di aplikasi
        self.image.texture = texture

class MyApp(App):
    def build(self):
        return AndroidCamera()

if __name__ == '__main__':
    MyApp().run()