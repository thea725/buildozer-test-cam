from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
import cv2
import numpy as np

class AndroidCamera(BoxLayout):
    def __init__(self, **kwargs):
        super(AndroidCamera, self).__init__(**kwargs)

        self.camera = Camera(play=True, index=0)
        self.camera.bind(on_tex=self.update_texture)
        self.add_widget(self.camera)

    def update_texture(self, instance, texture):
        frame = self.frame_from_buf()

        # Additional processing or modification of the frame can be done here

        self.frame_to_screen(frame)

        # Update the camera texture
        texture.blit_buffer(frame.tostring(), colorfmt='rgb', bufferfmt='ubyte')

    def frame_from_buf(self):
        w, h = self.camera.resolution
        frame = np.frombuffer(self.camera._camera._buffer.tostring(), dtype='uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
        return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        # Additional processing or modification of the frame can be done here

        # Example: Display the frame directly without additional processing
        pass

class MyApp(App):
    def build(self):
        return AndroidCamera()

if __name__ == '__main__':
    MyApp().run()