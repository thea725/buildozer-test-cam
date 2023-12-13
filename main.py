from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera

class CameraApp(App):
    def build(self):
        # Membuat tata letak utama
        layout = BoxLayout(orientation='vertical')

        # Menambahkan objek kamera
        self.camera = Camera(resolution=(640, 480), play=True)
        layout.add_widget(self.camera)

        return layout

if __name__ == '__main__':
    CameraApp().run()