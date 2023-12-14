from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.core.camera import Camera as CoreCamera
from kivy.properties import NumericProperty, ListProperty, BooleanProperty
import cv2
import numpy as np

class Camera(Image):
    play = BooleanProperty(False)
    index = NumericProperty(-1)
    resolution = ListProperty([-1, -1])

    def __init__(self, **kwargs):
        self._camera = None
        super(Camera, self).__init__(**kwargs)
        if self.index == -1:
            self.index = 0
        on_index = self._on_index
        fbind = self.fbind
        fbind('index', on_index)
        fbind('resolution', on_index)
        on_index()

    def on_tex(self, camera):
        frame = camera.frame
        if frame is not None:
            # Convert the frame to grayscale using OpenCV
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale frame to a Kivy texture
            texture = self._frame_to_texture(gray_frame)
            
            # Update the widget's texture and trigger a canvas update
            self.texture = texture
            self.texture_size = list(texture.size)
            self.canvas.ask_update()

    def _on_index(self, *largs):
        self._camera = None
        if self.index < 0:
            return
        if self.resolution[0] < 0 or self.resolution[1] < 0:
            self._camera = CoreCamera(index=self.index, stopped=True)
        else:
            self._camera = CoreCamera(index=self.index,
                                      resolution=self.resolution, stopped=True)
        if self.play:
            self._camera.start()

        self._camera.bind(on_texture=self.on_tex)

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()

    def _frame_to_texture(self, frame):
        # Convert the OpenCV frame to a Kivy texture
        texture = self._camera._frame_to_texture(frame)
        return texture



class AndroidCamera(BoxLayout):
    def __init__(self, **kwargs):
        super(AndroidCamera, self).__init__(**kwargs)

        self.camera = Camera(play=True)
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