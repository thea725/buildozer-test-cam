from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.core.camera import Camera as CoreCamera
from kivy.properties import NumericProperty, ListProperty, BooleanProperty
import cv2
import numpy as np
__all__ = ('Camera', )

def kivy_texture_to_numpy(texture):
    # Ensure that the texture has been loaded
    texture_size = texture.size
    texture.flip_vertical()  # Flip the texture vertically to match NumPy array orientation

    # Read the pixel data from the texture
    buffer = texture.pixels
    texture_data = np.frombuffer(buffer, dtype=np.uint8)

    # Reshape the data into a 3D NumPy array (height, width, channels)
    image_data = np.reshape(texture_data, (texture_size[1], texture_size[0], 4))

    # Extract RGB channels and ignore the alpha channel
    rgb_data = image_data[:, :, :3]

    return rgb_data

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
        frame = kivy_texture_to_numpy(camera.texture)
        # result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        texture = Texture.create(size=(result.shape[1], result.shape[0]), colorfmt='rgb')
        texture.blit_buffer(cv2.flip(result, 0).tobytes(), colorfmt='rgb', bufferfmt='ubyte')

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

class AndroidCamera(BoxLayout):
    def __init__(self, **kwargs):
        super(AndroidCamera, self).__init__(**kwargs)

        self.camera = Camera(play=True)
        # self.camera.bind(on_tex=self.update_texture)
        self.add_widget(self.camera)


class MyApp(App):
    def build(self):
        return AndroidCamera()

if __name__ == '__main__':
    MyApp().run()