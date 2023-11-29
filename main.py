from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
from kivy.clock import Clock
from imutils import perspective
import numpy as np
import imutils
class VideoApp(App):
    def build(self):
        self.pixelsPerMetric = 27.9394
        self.cap = cv2.VideoCapture('videos/vid4.mp4')  # Ganti dengan path video Anda
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        self.container = BoxLayout(orientation='vertical')
        self.image = Image()
        self.container.add_widget(self.image)

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update every 1/30th of a second

        return self.container

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            print("End of video.")
            return

        normal = self.defisheye(frame)
        enhance = self.normalization(normal)
        result = self.edge_detection(normal, enhance)

        frame = cv2.resize(result, (0, 0), fx=0.5, fy=0.5)
        frame_texture = self.texture_from_frame(frame)
        self.image.texture = frame_texture

    def defisheye(self, img):
        DIM = img.shape[:2][::-1]
        balance = 0
        K = np.array([[1122.4054962744387, 0.0, 1006.1145835723129], [0.0, 1129.0933478170655, 527.4670240270237], [0.0, 0.0, 1.0]])
        D = np.array([[-0.21503184621950375], [0.7441653867540186], [-1.3654196840660953], [0.8759864071569387]])
        dim1 = [1920, 1080]
        dim2 = [1920, 1080]
        dim3 = [1920, 1080]

        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance, fov_scale=1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

        dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return dst

    def normalization(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
        enhanced_image = clahe.apply(normalized_image)
        equ = cv2.equalizeHist(enhanced_image)

        upper_black = 75
        equ[np.where(equ <= upper_black)] = 0
        equ[np.where(equ <= upper_black)] = 0
        # blur = cv2.GaussianBlur(equ, (5, 5), 0)

        # # Deteksi tepi menggunakan Canny Edge Detection
        # edges = cv2.Canny(blur, 100, 200)

        # # Operasi morfologi untuk membersihkan dan memperjelas tepi
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # kernel = np.ones((5, 5), np.uint8)
        # dilated_edges = cv2.dilate(morph, kernel, iterations=1)
        # eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

        # # Gabungkan tepi dengan gambar asli
        # result = cv2.bitwise_and(img, img, mask=eroded_edges)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        return equ

    def edge_detection(self, frame, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 10  # Adjust this value as needed
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

        orig = frame.copy()
        for c in filtered_contours:
            # Measure the size of the object (contour) and calculate the scale factor
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

            # compute the Euclidean distance between the midpoints
            dA = self.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = self.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # compute the size of the object
            dimA = dA / self.pixelsPerMetric
            dimB = dB / self.pixelsPerMetric

            # if (dimA>5 and dimB>14) and (dimA<15 and dimB<37): #ukuran botol di antara 5-5 dan 14-37cm
            if dimA>1 and dimB>10 and (int(dimB/dimA) < 4 and int(dimB/dimA) > 2):
            # if True:
                print(dB)
                print("-",dimB)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                # draw the midpoints on the image
                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # draw lines between the midpoints
                cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                         (255, 0, 255), 2)
                cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                         (255, 0, 255), 2)

                # draw the object sizes on the image
                cv2.putText(orig, "{:.1f}cm".format(dimA),
                            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2)
                cv2.putText(orig, "{:.1f}cm".format(dimB),
                            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2)
        return orig
    
    def euclidean(self, point1, point2):
        # Mengonversi titik ke array NumPy untuk mendukung operasi vektor
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Menghitung jarak Euclidean
        distance = np.sqrt(np.sum((point1 - point2)**2))

        return distance
    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    def texture_from_frame(self, frame):
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

if __name__ == '__main__':
    VideoApp().run()
