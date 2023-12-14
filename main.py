from kivy.app import App
from kivy.uix.camera import Camera

class CameraApp(App):
    def build(self):
        return Camera()

if __name__ == '__main__':
    CameraApp().run()