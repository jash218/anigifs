from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog)
from PyQt6.QtCore import Qt, QTimer

from .audio import EnhancedAudioPlayer
from .renderer import MorphOpenGLWidget

class MainWindow(QMainWindow):
    """Main application window with 3D viewer and controls"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("3D Morph Animation")
        self.setGeometry(100, 100, 900, 700)
        self.audio_player = EnhancedAudioPlayer()
        
        # Create the OpenGL widget
        self.gl_widget = MorphOpenGLWidget(self)
        
        # Create control panel
        self.control_panel = QWidget()
        self.setup_control_panel()
        
        # Set up the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.gl_widget, 1)
        main_layout.addWidget(self.control_panel, 0)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Create update timer for synchronizing UI
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(50)  # 50ms is fast enough for UI updates
        
        # Ask for audio file on startup
        self.load_audio_file()
    
    def setup_control_panel(self):
        """Set up the control panel with sliders and buttons"""
        layout = QVBoxLayout(self.control_panel)
        layout.setContentsMargins(10, 0, 10, 10)
        
        # Progress label
        self.progress_label = QLabel("Progress: 0%")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: white; font-size: 14px;")
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #555555;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #AAAAAA;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.progress_slider.valueChanged.connect(self.slider_changed)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        
        # Rewind button
        self.rewind_btn = QPushButton("◀◀ Rewind")
        self.rewind_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.rewind_btn.clicked.connect(self.toggle_rewind)
        
        # Speed button
        self.speed_btn = QPushButton("1x")
        self.speed_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.speed_btn.clicked.connect(self.toggle_speed)
        
        # Audio button
        self.audio_btn = QPushButton("🔊 Audio")
        self.audio_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.audio_btn.clicked.connect(self.load_audio_file)
        
        # Time label
        self.time_label = QLabel("0:00 / 0:30")
        self.time_label.setStyleSheet("color: white; padding: 8px; font-size: 14px;")
        
        # Add controls to layout
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.rewind_btn)
        controls_layout.addWidget(self.audio_btn)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.speed_btn)
        
        # Add all elements to main layout
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_slider)
        layout.addLayout(controls_layout)
        
        # Style the control panel
        self.control_panel.setStyleSheet("background-color: #2D2D2D;")
        self.control_panel.setMaximumHeight(100)
        
        # Initialize state
        self.slider_being_dragged = False
    
    def slider_pressed(self):
        self.slider_being_dragged = True
    
    def slider_released(self):
        self.slider_being_dragged = False
        position_value = self.progress_slider.value()
        position_seconds = (position_value / 100.0) * self.gl_widget.duration
        self.gl_widget.set_position(position_seconds)
        self.audio_player.set_position(position_seconds)
    
    def slider_changed(self, value):
        if self.slider_being_dragged:
            position_seconds = (value / 100.0) * self.gl_widget.duration
            self.gl_widget.elapsed = position_seconds
            self.progress_label.setText(f"Progress: {value}%")
    
    def toggle_play_pause(self):
        self.gl_widget.toggle_play_pause()
        
        if self.gl_widget.paused:
            self.play_pause_btn.setText("▶ Play")
            if self.audio_player.playing:
                self.audio_player.pause()
        else:
            self.play_pause_btn.setText("⏸ Pause")
            self.rewind_btn.setText("◀◀ Rewind")
            self.gl_widget.rewind_mode = False
            self.audio_player.set_position(self.gl_widget.elapsed)
            self.audio_player.resume() if self.audio_player.paused else self.audio_player.play()
        
    def toggle_rewind(self):
        self.gl_widget.toggle_rewind()
        
        if self.gl_widget.rewind_mode:
            self.rewind_btn.setText("■ Stop")
            self.play_pause_btn.setText("▶ Play")
            self.audio_player.pause()
        else:
            self.rewind_btn.setText("◀◀ Rewind")
            self.play_pause_btn.setText("▶ Play")
            self.gl_widget.paused = True
    
    def toggle_speed(self):
        if self.gl_widget.speed == 1.0:
            self.gl_widget.speed = 2.0
            self.speed_btn.setText("2x")
        else:
            self.gl_widget.speed = 1.0
            self.speed_btn.setText("1x")
        
        self.audio_player.set_speed(self.gl_widget.speed)
    
    def update_ui(self):
        """Update UI elements based on current state"""
        # Only update slider if not being dragged
        if not self.slider_being_dragged:
            progress_percent = (self.gl_widget.elapsed / self.gl_widget.duration) * 100
            self.progress_slider.setValue(int(progress_percent))
            self.progress_label.setText(f"Progress: {int(progress_percent)}%")
        
        # Update time label
        mins_elapsed = int(self.gl_widget.elapsed) // 60
        secs_elapsed = int(self.gl_widget.elapsed) % 60
        mins_total = int(self.gl_widget.duration) // 60
        secs_total = int(self.gl_widget.duration) % 60
        self.time_label.setText(f"{mins_elapsed}:{secs_elapsed:02d} / {mins_total}:{secs_total:02d}")
    
    def update_player_controls(self):
        """Update player controls based on animation state"""
        if self.gl_widget.paused and not self.gl_widget.rewind_mode:
            self.play_pause_btn.setText("▶ Play")
        else:
            self.play_pause_btn.setText("⏸ Pause")
            
        if self.gl_widget.rewind_mode:
            self.rewind_btn.setText("■ Stop")
        else:
            self.rewind_btn.setText("◀◀ Rewind")
    
    def load_audio_file(self):
        """Load an audio file for playback"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select MP3 file for background music",
            "",
            "Audio files (*.mp3 *.wav);;All files (*)"
        )
        
        if file_name:
            # Stop any playing audio
            self.audio_player.cleanup()
            
            # Load the new file
            if self.audio_player.load_file(file_name):
                self.audio_btn.setText("🔊 Audio")
                if not self.gl_widget.paused:
                    self.audio_player.play(from_pos=self.gl_widget.elapsed)
            else:
                self.audio_btn.setText("🔇 No Audio")
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.audio_player.cleanup()
        event.accept()