from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.core.camera import Camera

class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.img = Image()
        self.layout.add_widget(self.img)

        # Buka kamera dengan mengakses indeks kamera (biasanya 0 untuk kamera belakang)
        self.camera = Camera(index=0, resolution=(640, 480))
        self.camera.play()

        Clock.schedule_interval(self.update_texture, 1 / 30.0)  # Update setiap 1/30 detik
        return self.layout

    def update_texture(self, dt):
        # Ambil frame dari kamera
        frame = self.camera.texture
        self.img.texture = frame

if __name__ == '__main__':
    CameraApp().run()
