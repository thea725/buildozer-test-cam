from kivy.app import App
from plyer import camera

class CameraApp(App):
    def build(self):
        camera.take_picture()

if __name__ == '__main__':
    CameraApp().run()
