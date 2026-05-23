import sys
import cv2

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QComboBox,
    QTextEdit,
    QMessageBox
)

from PyQt5.QtGui import (
    QPixmap,
    QImage
)

from PyQt5.QtCore import (
    Qt,
    QTimer
)

from predict import predict_image
from database import (
    save_history,
    get_history
)

# ========================
# MAIN WINDOW
# ========================
class WasteApp(QWidget):

    def __init__(self):

        super().__init__()

        self.image_path = ""

        # Camera
        self.cap = None

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_camera
        )

        self.setWindowTitle(
            "Smart Waste Classification"
        )

        self.setGeometry(
            200,
            100,
            700,
            900
        )

        self.init_ui()

    # ========================
    # UI
    # ========================
    def init_ui(self):

        layout = QVBoxLayout()

        # ========================
        # CAMERA / IMAGE LABEL
        # ========================
        self.image_label = QLabel(
            "No Image Selected"
        )

        self.image_label.setAlignment(
            Qt.AlignCenter
        )

        self.image_label.setFixedSize(
            500,
            400
        )

        layout.addWidget(
            self.image_label
        )

        # ========================
        # SELECT IMAGE BUTTON
        # ========================
        self.select_button = QPushButton(
            "Select Image"
        )

        self.select_button.clicked.connect(
            self.load_image
        )

        layout.addWidget(
            self.select_button
        )

        # ========================
        # OPEN CAMERA BUTTON
        # ========================
        self.camera_button = QPushButton(
            "Open Camera"
        )

        self.camera_button.clicked.connect(
            self.start_camera
        )

        layout.addWidget(
            self.camera_button
        )

        # ========================
        # STOP CAMERA BUTTON
        # ========================
        self.stop_button = QPushButton(
            "Stop Camera"
        )

        self.stop_button.clicked.connect(
            self.stop_camera
        )

        layout.addWidget(
            self.stop_button
        )

        # ========================
        # MODEL SELECT
        # ========================
        self.model_box = QComboBox()

        self.model_box.addItems([
            "Hybrid ViT + ResNet18",
            "EfficientNet"
        ])

        layout.addWidget(
            self.model_box
        )

        # ========================
        # PREDICT BUTTON
        # ========================
        self.predict_button = QPushButton(
            "Predict Image"
        )

        self.predict_button.clicked.connect(
            self.predict
        )

        layout.addWidget(
            self.predict_button
        )

        # ========================
        # RESULT LABEL
        # ========================
        self.result_label = QLabel(
            "Prediction Result"
        )

        self.result_label.setAlignment(
            Qt.AlignCenter
        )

        self.result_label.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            """
        )

        layout.addWidget(
            self.result_label
        )

        # ========================
        # HISTORY BOX
        # ========================
        self.history_box = QTextEdit()

        self.history_box.setReadOnly(True)

        layout.addWidget(
            self.history_box
        )

        self.setLayout(layout)

        self.load_history()

    # ========================
    # LOAD IMAGE
    # ========================
    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:

            self.image_path = file_path

            pixmap = QPixmap(file_path)

            pixmap = pixmap.scaled(
                500,
                400,
                Qt.KeepAspectRatio
            )

            self.image_label.setPixmap(
                pixmap
            )

    # ========================
    # PREDICT IMAGE
    # ========================
    def predict(self):

        if not self.image_path:

            QMessageBox.warning(
                self,
                "Warning",
                "Please select image first!"
            )

            return

        model_name = self.model_box.currentText()

        prediction, confidence = predict_image(
            self.image_path,
            model_name
        )

        result_text = (
            f"Detected Label: {prediction}\n"
            f"Confidence: {confidence:.2f}%"
        )

        self.result_label.setText(
            result_text
        )

        save_history(
            self.image_path,
            model_name,
            prediction,
            confidence
        )

        self.load_history()

    # ========================
    # START CAMERA
    # ========================
    def start_camera(self):

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            QMessageBox.warning(
                self,
                "Error",
                "Cannot open camera!"
            )

            return

        self.timer.start(30)

    # ========================
    # UPDATE CAMERA FRAME
    # ========================
    def update_camera(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        # ========================
        # SAVE TEMP FRAME
        # ========================
        temp_path = "temp_camera.jpg"

        cv2.imwrite(
            temp_path,
            frame
        )

        model_name = self.model_box.currentText()

        prediction, confidence = predict_image(
            temp_path,
            model_name
        )

        # ========================
        # SHOW RESULT
        # ========================
        result_text = (
            f"Detected Label: {prediction}\n"
            f"Confidence: {confidence:.2f}%"
        )

        self.result_label.setText(
            result_text
        )

        # ========================
        # DRAW TEXT ON CAMERA
        # ========================
        cv2.putText(
            frame,
            f"{prediction} ({confidence:.1f}%)",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # ========================
        # CONVERT FRAME
        # ========================
        rgb_image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb_image.shape

        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            qt_image
        )

        pixmap = pixmap.scaled(
            500,
            400,
            Qt.KeepAspectRatio
        )

        self.image_label.setPixmap(
            pixmap
        )

    # ========================
    # STOP CAMERA
    # ========================
    def stop_camera(self):

        self.timer.stop()

        if self.cap is not None:

            self.cap.release()

    # ========================
    # LOAD HISTORY
    # ========================
    def load_history(self):

        self.history_box.clear()

        histories = get_history()

        for row in histories:

            text = (
                f"Model: {row[2]} | "
                f"Result: {row[3]} | "
                f"Confidence: {row[4]:.2f}%"
            )

            self.history_box.append(text)

    # ========================
    # CLOSE EVENT
    # ========================
    def closeEvent(self, event):

        self.stop_camera()

        event.accept()


# ========================
# RUN APP
# ========================
app = QApplication(sys.argv)

window = WasteApp()

window.show()

sys.exit(app.exec_())