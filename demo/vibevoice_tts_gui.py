import sys
import os
import time
import queue
import traceback
import tempfile
import numpy as np
import soundfile as sf
import urllib.request
import tarfile
import shutil
from pathlib import Path
from typing import Optional, Dict

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QTextEdit, QComboBox,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox, QSplitter,
    QDialog, QListWidget, QListWidgetItem, QAbstractItemView
)
from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# Add project root and demo/web to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # demo/vibevoice_tts_gui.py -> demo/ -> root
web_dir = project_root / "demo" / "web"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(web_dir) not in sys.path:
    sys.path.insert(0, str(web_dir))

# Import the service from app.py
try:
    from app import StreamingTTSService
except ImportError as e:
    # If this fails, the user needs to make sure they are running from an environment
    # where the project structure is intact.
    raise ImportError(f"Could not import StreamingTTSService. Ensure you are in the project root or dependencies are installed. Error: {e}")

AVAILABLE_VOICE_PACKS = [
    {"name": "German Voices", "file": "experimental_voices_de.tar.gz", "url": "https://github.com/user-attachments/files/24035887/experimental_voices_de.tar.gz"},
    {"name": "French Voices", "file": "experimental_voices_fr.tar.gz", "url": "https://github.com/user-attachments/files/24035880/experimental_voices_fr.tar.gz"},
    {"name": "Japanese Voices", "file": "experimental_voices_jp.tar.gz", "url": "https://github.com/user-attachments/files/24035882/experimental_voices_jp.tar.gz"},
    {"name": "Korean Voices", "file": "experimental_voices_kr.tar.gz", "url": "https://github.com/user-attachments/files/24035883/experimental_voices_kr.tar.gz"},
    {"name": "Polish Voices", "file": "experimental_voices_pl.tar.gz", "url": "https://github.com/user-attachments/files/24035885/experimental_voices_pl.tar.gz"},
    {"name": "Portuguese Voices", "file": "experimental_voices_pt.tar.gz", "url": "https://github.com/user-attachments/files/24035886/experimental_voices_pt.tar.gz"},
    {"name": "Spanish Voices", "file": "experimental_voices_sp.tar.gz", "url": "https://github.com/user-attachments/files/24035884/experimental_voices_sp.tar.gz"},
    {"name": "English Voices 1", "file": "experimental_voices_en1.tar.gz", "url": "https://github.com/user-attachments/files/24189272/experimental_voices_en1.tar.gz"},
    {"name": "English Voices 2", "file": "experimental_voices_en2.tar.gz", "url": "https://github.com/user-attachments/files/24189273/experimental_voices_en2.tar.gz"},
]

class VoiceDownloadWorker(QThread):
    progress = Signal(float)  # Percentage
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, voice_pack, target_dir):
        super().__init__()
        self.voice_pack = voice_pack
        self.target_dir = target_dir
        self._is_running = True

    def run(self):
        try:
            url = self.voice_pack["url"]
            filename = self.voice_pack["file"]
            filepath = os.path.join(self.target_dir, filename)

            # Ensure target directory exists
            os.makedirs(self.target_dir, exist_ok=True)

            self.log.emit(f"Downloading {filename}...")

            def report_hook(block_num, block_size, total_size):
                if not self._is_running:
                    raise InterruptedError("Download cancelled")
                if total_size > 0:
                    percent = (block_num * block_size * 100) / total_size
                    self.progress.emit(percent)

            if not os.path.exists(filepath):
                 urllib.request.urlretrieve(url, filepath, report_hook)
            else:
                 self.log.emit(f"{filename} already exists, skipping download.")

            self.log.emit(f"Extracting {filename}...")
            self.progress.emit(0) # Reset progress for extraction (indeterminate mostly)

            with tarfile.open(filepath, "r:gz") as tar:
                # tar.extractall(path=self.target_dir) # This can be slow and blocking without progress
                # Iterate members to show some activity
                members = tar.getmembers()
                total_members = len(members)
                for i, member in enumerate(members):
                     if not self._is_running:
                         break
                     tar.extract(member, path=self.target_dir)
                     if total_members > 0:
                         self.progress.emit((i / total_members) * 100)

            # Clean up tar file
            if os.path.exists(filepath):
                os.remove(filepath)

            self.log.emit("Done.")
            self.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False

class VoiceDownloaderDialog(QDialog):
    voices_downloaded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Experimental Voices")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Search/Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter voices...")
        self.search_input.textChanged.connect(self.filter_voices)
        filter_layout.addWidget(self.search_input)
        layout.addLayout(filter_layout)

        # List of voices
        self.voice_list = QListWidget()
        self.voice_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.voice_list)

        self.populate_list()

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.download_btn = QPushButton("Download Selected")
        self.download_btn.clicked.connect(self.download_selected)
        btn_layout.addWidget(self.download_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        self.worker = None

    def populate_list(self):
        self.voice_list.clear()
        for pack in AVAILABLE_VOICE_PACKS:
            item = QListWidgetItem(pack["name"])
            item.setData(Qt.UserRole, pack)
            self.voice_list.addItem(item)

    def filter_voices(self, text):
        for i in range(self.voice_list.count()):
            item = self.voice_list.item(i)
            pack = item.data(Qt.UserRole)
            if text.lower() in pack["name"].lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def download_selected(self):
        items = self.voice_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Warning", "Please select a voice pack to download.")
            return

        item = items[0]
        pack = item.data(Qt.UserRole)

        target_dir = str(project_root / "demo" / "voices" / "streaming_model" / "experimental_voices")

        self.download_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.voice_list.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Starting download of {pack['name']}...")

        self.worker = VoiceDownloadWorker(pack, target_dir)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_download_finished)
        self.worker.error.connect(self.on_download_error)
        self.worker.start()

    def on_download_finished(self):
        self.status_label.setText("Download completed successfully!")
        self.download_btn.setEnabled(True)
        self.close_btn.setEnabled(True)
        self.voice_list.setEnabled(True)
        self.progress_bar.hide()
        self.worker = None
        self.voices_downloaded.emit()
        QMessageBox.information(self, "Success", "Voice pack installed!")

    def on_download_error(self, error):
        self.status_label.setText(f"Error: {error}")
        self.download_btn.setEnabled(True)
        self.close_btn.setEnabled(True)
        self.voice_list.setEnabled(True)
        self.progress_bar.hide()
        self.worker = None
        QMessageBox.critical(self, "Error", f"Download failed:\n{error}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

class TTSWorker(QThread):
    """
    Worker thread to handle TTS generation to avoid freezing the GUI.
    """
    progress = Signal(float)  # Percentage or time
    finished = Signal(str)    # Path to generated file
    error = Signal(str)       # Error message
    log = Signal(str)         # Log messages

    def __init__(self, service: StreamingTTSService, text: str, params: dict):
        super().__init__()
        self.service = service
        self.text = text
        self.params = params
        self._is_running = True

    def run(self):
        try:
            self.log.emit("Starting generation...")

            # Prepare to collect audio chunks
            audio_chunks = []

            # Call the streaming generator
            # We iterate over the generator to consume chunks
            stream = self.service.stream(
                text=self.text,
                voice_key=self.params.get("voice_key"),
                cfg_scale=self.params.get("cfg_scale", 1.5),
                inference_steps=self.params.get("inference_steps", 5),
                do_sample=self.params.get("do_sample", False),
                temperature=self.params.get("temperature", 0.9),
                top_p=self.params.get("top_p", 0.9),
                # We don't pass log_callback here as it's for the websocket
                # But we can update progress based on chunks received if we knew total length
            )

            for chunk in stream:
                if not self._is_running:
                    break
                audio_chunks.append(chunk)
                # Emit some progress or log?
                # Since we don't know total duration easily ahead of time without pre-calc,
                # we just show activity.
                self.progress.emit(-1) # Indeterminate

            if not self._is_running:
                return

            if not audio_chunks:
                self.error.emit("No audio generated.")
                return

            # Concatenate chunks
            full_audio = np.concatenate(audio_chunks)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_path = tf.name

            # Save using soundfile
            sf.write(temp_path, full_audio, self.service.sample_rate)

            self.log.emit(f"Generation complete. Saved to {temp_path}")
            self.finished.emit(temp_path)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False


class ModelLoaderWorker(QThread):
    """
    Worker to load the model in background.
    """
    finished = Signal()
    error = Signal(str)

    def __init__(self, model_path: str, device: str):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.service = None

    def run(self):
        try:
            self.service = StreamingTTSService(
                model_path=self.model_path,
                device=self.device
            )
            self.service.load()
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class VibeVoiceTTSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VibeVoice TTS GUI")
        self.resize(900, 700)

        self.service: Optional[StreamingTTSService] = None
        self.model_loader: Optional[ModelLoaderWorker] = None
        self.tts_worker: Optional[TTSWorker] = None
        self.temp_audio_path: Optional[str] = None

        # Audio Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Model Configuration Section ---
        model_group = QGroupBox("Model Configuration")
        model_layout = QHBoxLayout()

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Model Path (e.g. microsoft/VibeVoice-Realtime-0.5B)")
        self.path_input.setText(os.environ.get("MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B"))

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu", "mps"])
        if os.environ.get("MODEL_DEVICE"):
            index = self.device_combo.findText(os.environ.get("MODEL_DEVICE"))
            if index >= 0: self.device_combo.setCurrentIndex(index)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)

        self.model_status_label = QLabel("Not Loaded")
        self.model_status_label.setStyleSheet("color: red; font-weight: bold;")

        model_layout.addWidget(QLabel("Path:"))
        model_layout.addWidget(self.path_input, 2)
        model_layout.addWidget(QLabel("Device:"))
        model_layout.addWidget(self.device_combo)
        model_layout.addWidget(self.load_btn)
        model_layout.addWidget(self.model_status_label)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # --- Main Content Splitter ---
        splitter = QSplitter(Qt.Vertical)

        # --- Input Section ---
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)

        input_layout.addWidget(QLabel("Input Text:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Type your text here...")
        input_layout.addWidget(self.text_input)

        splitter.addWidget(input_widget)

        # --- Settings & Control Section ---
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        # Parameters
        params_group = QGroupBox("Generation Parameters")
        params_form = QFormLayout()

        # Voice
        voice_layout = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setToolTip("Select the voice persona.")
        voice_layout.addWidget(self.voice_combo, 1)

        self.download_voices_btn = QPushButton("Download Voices")
        self.download_voices_btn.setToolTip("Search and download experimental voices.")
        self.download_voices_btn.clicked.connect(self.open_voice_downloader)
        voice_layout.addWidget(self.download_voices_btn)

        params_form.addRow("Voice:", voice_layout)

        # CFG Scale
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(0.1, 10.0)
        self.cfg_spin.setValue(1.5)
        self.cfg_spin.setSingleStep(0.1)
        self.cfg_spin.setToolTip("Classifier Free Guidance scale.\nControls how closely the audio follows the text.\nHigher values (e.g., 1.5) mean stronger adherence.")
        params_form.addRow("CFG Scale:", self.cfg_spin)

        # Inference Steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(5)
        self.steps_spin.setToolTip("Number of denoising steps.\nHigher values might improve quality but take longer.\nDefault is typically 5.")
        params_form.addRow("Inference Steps:", self.steps_spin)

        # Do Sample
        self.sample_check = QCheckBox()
        self.sample_check.setChecked(False)
        self.sample_check.toggled.connect(self.toggle_sampling_params)
        self.sample_check.setToolTip("Enable random sampling.\nIf unchecked, generation is deterministic.")
        params_form.addRow("Do Sample:", self.sample_check)

        # Temperature
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setValue(0.9)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setEnabled(False)
        self.temp_spin.setToolTip("Randomness in generation.\nLower values make it more deterministic, higher values more expressive.\nOnly active if 'Do Sample' is checked.")
        params_form.addRow("Temperature:", self.temp_spin)

        # Top P
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setEnabled(False)
        self.top_p_spin.setToolTip("Nucleus sampling probability.\nLimits the generated tokens to the top P probability mass.\nOnly active if 'Do Sample' is checked.")
        params_form.addRow("Top P:", self.top_p_spin)

        params_group.setLayout(params_form)
        controls_layout.addWidget(params_group)

        # Actions
        actions_layout = QVBoxLayout()
        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.setMinimumHeight(50)
        self.generate_btn.clicked.connect(self.start_generation)
        self.generate_btn.setEnabled(False)
        actions_layout.addWidget(self.generate_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate by default during gen
        self.progress_bar.hide()
        actions_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        actions_layout.addWidget(self.status_label)

        # Playback Controls
        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)

        self.save_btn = QPushButton("Save As...")
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)

        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addWidget(self.save_btn)
        playback_group.setLayout(playback_layout)

        actions_layout.addWidget(playback_group)
        actions_layout.addStretch()

        controls_layout.addLayout(actions_layout)

        splitter.addWidget(controls_widget)
        main_layout.addWidget(splitter)

    def toggle_sampling_params(self, checked):
        self.temp_spin.setEnabled(checked)
        self.top_p_spin.setEnabled(checked)

    def open_voice_downloader(self):
        dialog = VoiceDownloaderDialog(self)
        dialog.voices_downloaded.connect(self.refresh_voices)
        dialog.exec()

    def refresh_voices(self):
        if self.service:
            try:
                self.service.reload_voices()
                current_voice = self.voice_combo.currentText()
                self.voice_combo.clear()
                if self.service.voice_presets:
                    voices = sorted(self.service.voice_presets.keys())
                    self.voice_combo.addItems(voices)

                    if current_voice in voices:
                         index = self.voice_combo.findText(current_voice)
                         self.voice_combo.setCurrentIndex(index)
                    elif self.service.default_voice_key:
                        index = self.voice_combo.findText(self.service.default_voice_key)
                        if index >= 0: self.voice_combo.setCurrentIndex(index)
                QMessageBox.information(self, "Voices Reloaded", "Voice list has been updated.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to reload voices: {e}")

    def load_model(self):
        path = self.path_input.text().strip()
        device = self.device_combo.currentText()
        if not path:
            QMessageBox.warning(self, "Error", "Please enter a model path.")
            return

        self.load_btn.setEnabled(False)
        self.path_input.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.model_status_label.setText("Loading Model... (this may take a while)")
        self.model_status_label.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()

        self.model_loader = ModelLoaderWorker(path, device)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.on_model_load_error)
        self.model_loader.start()

    def on_model_loaded(self):
        self.service = self.model_loader.service

        # Populate voices
        self.voice_combo.clear()
        if self.service.voice_presets:
            voices = sorted(self.service.voice_presets.keys())
            self.voice_combo.addItems(voices)
            if self.service.default_voice_key:
                index = self.voice_combo.findText(self.service.default_voice_key)
                if index >= 0: self.voice_combo.setCurrentIndex(index)

        self.model_status_label.setText("Model Loaded")
        self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.generate_btn.setEnabled(True)
        self.load_btn.setText("Reload Model")
        self.load_btn.setEnabled(True)
        self.path_input.setEnabled(True)
        self.device_combo.setEnabled(True)
        QMessageBox.information(self, "Success", "Model loaded successfully!")

    def on_model_load_error(self, err_msg):
        self.model_status_label.setText("Load Failed")
        self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.load_btn.setEnabled(True)
        self.path_input.setEnabled(True)
        self.device_combo.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to load model:\n{err_msg}")

    def start_generation(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter some text.")
            return

        # Disable controls
        self.generate_btn.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText("Generating...")

        params = {
            "voice_key": self.voice_combo.currentText(),
            "cfg_scale": self.cfg_spin.value(),
            "inference_steps": self.steps_spin.value(),
            "do_sample": self.sample_check.isChecked(),
            "temperature": self.temp_spin.value(),
            "top_p": self.top_p_spin.value(),
        }

        self.tts_worker = TTSWorker(self.service, text, params)
        self.tts_worker.finished.connect(self.on_generation_finished)
        self.tts_worker.error.connect(self.on_generation_error)
        self.tts_worker.log.connect(lambda msg: self.status_label.setText(msg))
        self.tts_worker.start()

    def on_generation_finished(self, file_path):
        self.temp_audio_path = file_path
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Generation Complete")

        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        # Load into player
        self.player.setSource(QUrl.fromLocalFile(file_path))

    def on_generation_error(self, err_msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Generation Error", f"An error occurred:\n{err_msg}")

    def play_audio(self):
        if self.temp_audio_path:
            self.player.play()

    def stop_audio(self):
        self.player.stop()

    def save_audio(self):
        if not self.temp_audio_path:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", str(Path.home()), "WAV Files (*.wav)"
        )
        if file_path:
            try:
                shutil.copy2(self.temp_audio_path, file_path)
                QMessageBox.information(self, "Success", f"Saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")

    def closeEvent(self, event):
        # Cleanup temp file if exists
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            try:
                os.remove(self.temp_audio_path)
            except:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VibeVoiceTTSWindow()
    window.show()
    sys.exit(app.exec())
