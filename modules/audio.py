import sys
import os
import time
import math
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QSplitter)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import librosa
import sounddevice as sd


class SmoothValue:
    def __init__(self, initial=0.0, smoothing=0.8):
        self.current = initial
        self.target = initial
        self.smoothing = smoothing
    
    def update(self):
        self.current += (self.target - self.current) * (1.0 - self.smoothing)
    
    def set_target(self, value):
        self.target = value
    
    def get(self):
        return self.current


class RotationController:
    def __init__(self):
        self.rotation_x = SmoothValue()
        self.rotation_y = SmoothValue()
        self.momentum_x = 0.0
        self.momentum_y = 0.0
        self.velocity_x = SmoothValue(smoothing=0.6)
        self.velocity_y = SmoothValue(smoothing=0.6)
        self.auto_rotation = True
        self.transition = SmoothValue(initial=1.0, smoothing=0.95)
        self.damping = 0.95
        self.momentum_multiplier = 0.2
        self.last_time = time.time()
        
    def update(self, delta_time):
        # Update smooth values
        self.rotation_x.update()
        self.rotation_y.update()
        self.velocity_x.update()
        self.velocity_y.update()
        self.transition.update()
        
        # Apply momentum
        if abs(self.momentum_x) > 0.01 or abs(self.momentum_y) > 0.01:
            self.rotation_x.set_target(self.rotation_x.get() + self.momentum_x)
            self.rotation_y.set_target(self.rotation_y.get() + self.momentum_y)
            
            # Modified dynamic damping for smoother deceleration
            velocity = (abs(self.momentum_x) + abs(self.momentum_y)) * 0.5
            dynamic_damping = self.damping * (0.97 + 0.03 * (1.0 - min(1.0, velocity / 3.0)))
            
            self.momentum_x *= dynamic_damping
            self.momentum_y *= dynamic_damping
        
        # Clear tiny momentum values
        if abs(self.momentum_x) < 0.01: self.momentum_x = 0
        if abs(self.momentum_y) < 0.01: self.momentum_y = 0
    
    def start_orbit(self):
        self.auto_rotation = False
        self.transition.set_target(0.0)
        
    def stop_orbit(self):
        self.momentum_x = self.velocity_y.get() * self.momentum_multiplier
        self.momentum_y = self.velocity_x.get() * self.momentum_multiplier
        self.transition.set_target(1.0)
        self.auto_rotation = True
        
    def should_resume_auto_rotation(self):
        return (abs(self.momentum_x) < 0.3 and
                abs(self.momentum_y) < 0.3 and
                self.transition.get() > 0.95)
    
    def resume_auto_rotation(self):
        self.auto_rotation = True
        self.transition.set_target(1.0)
    
    def update_mouse_velocity(self, dx, dy, delta_time):
        if delta_time > 0:
            self.velocity_x.set_target(dx / delta_time * 1.3)
            self.velocity_y.set_target(dy / delta_time * 1.3)


class EnhancedAudioPlayer:
    def __init__(self):
        self.audio_data = None
        self.sample_rate = None
        self.playing = False
        self.paused = False
        self.current_position = 0.0
        self.speed = 1.0
        self.is_rewinding = False

    def load_file(self, file_path):
        try:
            data, sr = librosa.load(file_path, sr=None, mono=True)
            self.audio_data = data.astype(np.float32)
            self.sample_rate = sr
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

    def play(self, from_pos=None):
        if self.audio_data is None:
            return
        if from_pos is not None:
            self.current_position = from_pos
        start_sample = int(self.current_position * self.sample_rate)
        sd.stop()
        if start_sample < len(self.audio_data):
            chunk = self.audio_data[start_sample:]
            sd.play(chunk, samplerate=int(self.sample_rate * self.speed))
            self.playing = True
            self.paused = False

    def pause(self):
        if self.playing and not self.paused:
            try:
                stream = sd.get_stream()
                if stream is not None:
                    elapsed = stream.time
                    self.current_position += elapsed / self.speed
            except:
                pass  # Handle case when stream is not available
            sd.stop()
            self.paused = True

    def resume(self):
        if self.paused:
            self.play(from_pos=self.current_position)

    def set_position(self, position_seconds):
        if self.audio_data is not None:
            self.current_position = max(0.0, min(position_seconds, 
                                               len(self.audio_data) / self.sample_rate))
            if self.playing and not self.paused:
                self.play(from_pos=self.current_position)

    def set_speed(self, speed):
        self.speed = speed
        if self.playing and not self.paused:
            self.play(from_pos=self.current_position)

    def start_rewind(self):
        self.is_rewinding = True
        sd.stop()
        self.playing = False

    def stop_rewind(self):
        self.is_rewinding = False

    def cleanup(self):
        sd.stop()
        self.playing = False
        self.paused = False
        self.is_rewinding = False


class MorphOpenGLWidget(QOpenGLWidget):
    """OpenGL Widget for morphing cube to sphere"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.controller = RotationController()
        self.last_mouse_pos = None
        self.last_time = time.time()
        
        # Animation properties
        self.elapsed = 0.0
        self.duration = 30.0
        self.paused = True
        self.rewind_mode = False
        self.speed = 1.0
        
        # Rendering properties
        self.cube_vertices = None
        self.sphere_vertices = None
        self.edges = None
        self.subdivisions = 5
        
        # Set up the timer for animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)  # ~60 FPS
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Set minimum size for the widget
        self.setMinimumSize(800, 600)
        
    def initializeGL(self):
        """Initialize OpenGL and prepare rendering resources"""
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.1, 1.0)
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create geometry
        self._create_geometry()
        
    def resizeGL(self, w, h):
        """Handle window resize events"""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """Render the scene"""
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset view
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)
        
        # Apply rotations
        glRotatef(self.controller.rotation_x.get(), 1, 0, 0)
        glRotatef(self.controller.rotation_y.get(), 0, 1, 0)
        
        # Auto-rotation
        t = self.get_normalized_elapsed()
        if self.controller.auto_rotation:
            rotation_speed = 50 * (1.0 - 0.5 * t)
            rotation_angle = rotation_speed * self.elapsed
            glRotatef(rotation_angle % 360, 1, 1, 1)
        
        # Draw the morphed shape
        self._draw_morphed_shape(t)
        
        # Draw axis indicator
        self._draw_axis_indicator(t)
        
    def _create_geometry(self):
        """Create the tessellated cube and sphere geometries"""
        self.cube_vertices, self.edges = self._create_tessellated_cube(size=1.0, subdivisions=self.subdivisions)
        self.sphere_vertices = self._generate_sphere_vertices(self.cube_vertices)
        
    def _create_tessellated_cube(self, size=1.0, subdivisions=3):
        """Generate a tessellated cube with the given subdivisions"""
        vertices = []
        edges = []
        
        # Basic cube vertices and faces
        basic_vertices = [
            [-size, -size, -size],  # 0: bottom left back
            [size, -size, -size],   # 1: bottom right back
            [size, size, -size],    # 2: top right back
            [-size, size, -size],   # 3: top left back
            [-size, -size, size],   # 4: bottom left front
            [size, -size, size],    # 5: bottom right front
            [size, size, size],     # 6: top right front
            [-size, size, size]     # 7: top left front
        ]
        
        basic_quads = [
            (0, 3, 2, 1),  # Back face
            (4, 5, 6, 7),  # Front face
            (0, 1, 5, 4),  # Bottom face
            (3, 7, 6, 2),  # Top face
            (0, 4, 7, 3),  # Left face
            (1, 2, 6, 5)   # Right face
        ]
        
        # Vertex cache for deduplication
        vertex_cache = {}
        
        # Process each face
        for quad in basic_quads:
            corners = [basic_vertices[i] for i in quad]
            grid_indices = []
            
            for i in range(subdivisions + 1):
                row_indices = []
                t1 = i / subdivisions
                
                for j in range(subdivisions + 1):
                    t2 = j / subdivisions
                    
                    # Interpolate along edges
                    edge1 = self._interpolate(corners[0], corners[1], t2)
                    edge2 = self._interpolate(corners[3], corners[2], t2)
                    
                    # Interpolate between edges
                    point = self._interpolate(edge1, edge2, t1)
                    
                    # Add vertex with deduplication
                    point_tuple = tuple(map(lambda x: round(x, 6), point))
                    if point_tuple in vertex_cache:
                        idx = vertex_cache[point_tuple]
                    else:
                        vertices.append(point)
                        idx = len(vertices) - 1
                        vertex_cache[point_tuple] = idx
                        
                    row_indices.append(idx)
                
                grid_indices.append(row_indices)
            
            # Create edges
            for i in range(subdivisions + 1):
                for j in range(subdivisions):
                    # Horizontal edge
                    edges.append((grid_indices[i][j], grid_indices[i][j+1]))
                    
                    # Vertical edge
                    if i < subdivisions:
                        edges.append((grid_indices[i][j], grid_indices[i+1][j]))
        
        return np.array(vertices, dtype=np.float32), edges
        
    def _generate_sphere_vertices(self, cube_vertices, radius=1.0):
        """Generate sphere vertices that correspond to cube vertices"""
        sphere_vertices = []
        
        for v in cube_vertices:
            # Normalize to sphere radius
            distance = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            
            if distance < 0.0001:
                sphere_vertices.append([0, 0, 0])
            else:
                sphere_vertices.append([
                    v[0] / distance * radius,
                    v[1] / distance * radius,
                    v[2] / distance * radius
                ])
                
        return np.array(sphere_vertices, dtype=np.float32)
    
    def _interpolate(self, v1, v2, t):
        """Linear interpolation helper"""
        return [v1[0] + (v2[0] - v1[0]) * t,
                v1[1] + (v2[1] - v1[1]) * t,
                v1[2] + (v2[2] - v1[2]) * t]
    
    def _lerp(self, start, end, t):
        """Linear interpolation between arrays"""
        return start + (end - start) * t
    
    def _draw_morphed_shape(self, t):
        """Draw the morphed shape with optimized rendering"""
        # Calculate morphed vertices
        morphed_vertices = self._lerp(self.cube_vertices, self.sphere_vertices, t)
        
        # Draw edges efficiently
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for edge in self.edges:
            for vertex_idx in edge:
                glVertex3fv(morphed_vertices[vertex_idx])
        glEnd()
        
        # Add sphere overlay when t > 0.7
        if t > 0.7:
            opacity = (t - 0.7) / 0.3
            glColor4f(0.5, 0.5, 1.0, opacity * 0.5)
            
            quadric = gluNewQuadric()
            gluQuadricDrawStyle(quadric, GLU_LINE)
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, 1.0, 20, 20)
            gluDeleteQuadric(quadric)
    
    def _draw_axis_indicator(self, t):
        """Draw XYZ axis indicator in the top-left corner"""
        # Save current matrices and viewport
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        # Get viewport
        viewport = glGetIntegerv(GL_VIEWPORT)
        indicator_size = 80
        
        # Set indicator viewport
        glViewport(10, viewport[3] - indicator_size - 10, indicator_size, indicator_size)
        
        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1.0, 0.1, 50.0)
        
        # Setup modelview
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)
        
        # Apply rotations
        glRotatef(self.controller.rotation_x.get(), 1, 0, 0)
        glRotatef(self.controller.rotation_y.get(), 0, 1, 0)
        
        # Auto-rotation
        if self.controller.auto_rotation:
            rotation_speed = 50 * (1.0 - 0.5 * t)
            rotation_angle = rotation_speed * self.elapsed
            glRotatef(rotation_angle % 360, 1, 1, 1)
        
        # Draw axes
        glLineWidth(2.0)
        glBegin(GL_LINES)
        
        # X-axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        
        # Y-axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        
        # Z-axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        
        glEnd()
        
        # Draw direction cones
        # X-axis cone
        self._draw_axis_cone(1.0, 0.0, 0.0, 90.0, 0.0, 1.0, 0.0, (1.0, 0.0, 0.0))
        
        # Y-axis cone
        self._draw_axis_cone(0.0, 1.0, 0.0, -90.0, 1.0, 0.0, 0.0, (0.0, 1.0, 0.0))
        
        # Z-axis cone
        self._draw_axis_cone(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 1.0))
        
        # Restore viewport and matrices
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _draw_axis_cone(self, x, y, z, angle, rx, ry, rz, color):
        """Helper to draw cone at the end of an axis"""
        glColor3fv(color)
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(angle, rx, ry, rz)
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_FILL)
        gluCylinder(quadric, 0.0, 0.1, 0.2, 8, 1)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()
            self.controller.start_orbit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
            self.controller.stop_orbit()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.last_mouse_pos is not None:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time
            
            self.controller.update_mouse_velocity(dx, dy, delta_time)
            self.controller.rotation_y.set_target(
                self.controller.rotation_y.get() + dx * 0.5)
            self.controller.rotation_x.set_target(
                self.controller.rotation_x.get() + dy * 0.5)
            
            self.last_mouse_pos = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key.Key_Space:
            if self.controller.auto_rotation:
                self.controller.start_orbit()
            else:
                self.controller.resume_auto_rotation()
            self.update()
        elif event.key() == Qt.Key.Key_Escape:
            QApplication.quit()
        super().keyPressEvent(event)

    def update_animation(self):
        """Update animation state"""
        # Update controller
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        
        self.controller.update(delta_time)
        
        # Check auto-rotation
        if not self.controller.auto_rotation and self.controller.should_resume_auto_rotation():
            self.controller.resume_auto_rotation()
        
        # Update elapsed time based on play state
        if not self.paused:
            if self.rewind_mode:
                self.elapsed -= 0.016 * 2.0
                if self.elapsed < 0:
                    self.elapsed = 0
                    self.rewind_mode = False
                    self.paused = True
                    if self.parentWidget():
                        self.parentWidget().update_player_controls()
            else:
                self.elapsed += 0.016 * self.speed
                if self.elapsed > self.duration:
                    self.elapsed = self.duration
                    self.paused = True
                    if self.parentWidget():
                        self.parentWidget().update_player_controls()
        
        # Request update
        self.update()
    
    def get_normalized_elapsed(self):
        """Get normalized elapsed time (0.0 to 1.0)"""
        return min(1.0, max(0.0, self.elapsed / self.duration))
    
    def set_position(self, position_seconds):
        """Set the animation position"""
        self.elapsed = max(0.0, min(position_seconds, self.duration))
        self.update()

    def toggle_play_pause(self):
        """Toggle playback state"""
        self.rewind_mode = False
        self.paused = not self.paused

    def toggle_rewind(self):
        """Toggle rewind mode"""
        if self.rewind_mode:
            self.rewind_mode = False
            self.paused = True
        else:
            self.rewind_mode = True
            self.paused = False

    def set_speed(self, speed):
        """Set playback speed"""
        self.speed = speed


class MainWindow(QMainWindow):
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


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Set up OpenGL format
    gl_format = QSurfaceFormat()
    gl_format.setVersion(2, 1)  # Using OpenGL 2.1 for compatibility
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    gl_format.setSamples(4)  # 4x MSAA
    QSurfaceFormat.setDefaultFormat(gl_format)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()