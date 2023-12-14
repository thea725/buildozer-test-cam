import cv2
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock

class CameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0,cv2.CAP_ANDROID)
        self.img = Image()

        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.img

    def update(self, dt):
        ret, frame = self.capture.read()
        self.img.texture = self.texture(frame)

    def texture(self, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return image_texture

if __name__ == '__main__':
    CameraApp().run()
