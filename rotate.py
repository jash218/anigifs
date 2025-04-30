import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import sys
import math
import os
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QLabel)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QFont
import librosa
import soundfile as sf
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QFileDialog



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
        # Convert current velocity to momentum
        self.momentum_x = self.velocity_y.get() * self.momentum_multiplier
        self.momentum_y = self.velocity_x.get() * self.momentum_multiplier

        # start winding transition back up to auto‐rotation
        self.transition.set_target(1.0)

        # re‐enable auto‐rotation immediately (it will fade in smoothly)
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
            # Increased mouse sensitivity for faster response during interaction
            self.velocity_x.set_target(dx / delta_time * 1.3)
            self.velocity_y.set_target(dy / delta_time * 1.3)

class EnhancedAudioPlayer:
    def __init__(self):
        self.audio_data       = None
        self.sample_rate      = None
        self.playing          = False
        self.paused           = False
        self.current_position = 0.0   # in seconds
        self.speed            = 1.0
        self.is_rewinding     = False

    def load_file(self, file_path):
        data, sr = librosa.load(file_path, sr=None, mono=True)
        self.audio_data  = data.astype(np.float32)
        self.sample_rate = sr

    def play(self, from_pos=None):
         if self.audio_data is None:
             return
         if from_pos is not None:
             self.current_position = from_pos
         start_sample = int(self.current_position * self.sample_rate)
         sd.stop()
         chunk = self.audio_data[start_sample:]
   # sounddevice.play only needs data + samplerate; it uses chunk.dtype
         sd.play(chunk, samplerate=int(self.sample_rate * self.speed))
         self.playing = True
         self.paused  = False


    def pause(self):
        if self.playing and not self.paused:
            # sounddevice doesn’t have a built-in pause – we stop and record where we were
            elapsed = sd.get_stream().time
            self.current_position += elapsed / self.speed
            sd.stop()
            self.paused = True

    def resume(self):
        if self.paused:
            self.play(from_pos=self.current_position)

    def set_position(self, position_seconds):
        self.current_position = max(0.0,
            min(position_seconds, len(self.audio_data) / self.sample_rate))
        if self.playing and not self.paused:
            self.play(from_pos=self.current_position)

    def set_speed(self, speed):
        self.speed = speed
        if self.playing and not self.paused:
            self.play(from_pos=self.current_position)

    # **Stubs for rewind support** so your AnimationThread no longer crashes:
    def start_rewind(self):
        self.is_rewinding = True
        # (if you want real rewind, you'd need to implement reversed playback here)

    def stop_rewind(self):
        self.is_rewinding = False
        # (and maybe resume normal play at current_position)

    def cleanup(self):
        sd.stop()
        self.playing = False
        self.paused  = False
        self.is_rewinding = False



class ProgressWindow(QMainWindow):
    # Signal to notify the animation thread of changes
    animation_signal = pyqtSignal(dict)
    
    def __init__(self, duration=30.0):
        super().__init__()
        
        # Initialize variables
        self.duration = duration
        self.elapsed = 0
        self.running = True
        self.paused = True
        self.speed = 1.0
        self.start_animation = False
        self.rewind_mode = False
        
        # Setup UI
        self.init_ui()
        
        # Start the timer for UI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(16)  # ~60 fps for UI updates
        
        # Show the window
        self.show()
    
    def init_ui(self):
        # Main window setup
        self.setWindowTitle("Animation Control")
        self.setGeometry(100, 100, 500, 150)
        self.setStyleSheet("background-color: #2D2D2D; color: white;")
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
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
        
        # Progress label
        self.progress_label = QLabel("Progress: 0%")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                border: none;
                padding: 8px;
                border-radius: 4px;
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
                border: none;
                padding: 8px;
                border-radius: 4px;
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
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.speed_btn.clicked.connect(self.toggle_speed)
        
        # Time label
        self.time_label = QLabel("0:00 / 0:30")
        self.time_label.setStyleSheet("padding: 8px;")
        
        # Add widgets to layouts
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.rewind_btn)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.speed_btn)
        
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.progress_slider)
        main_layout.addLayout(controls_layout)
        
        # Set a fixed size
        self.setFixedSize(500, 150)
        
        # Connect close event
        self.slider_being_dragged = False
    
    def closeEvent(self, event):
        self.running = False
        event.accept()
        # Force terminate the program
        os._exit(0)
    
    def update_ui(self):
        if self.running:
            # Update progress if not paused
            if not self.paused:
                if self.rewind_mode:
                    # Rewind at 2x speed
                    self.elapsed -= 0.016 * 2.0  # 60fps * 2x speed
                    if self.elapsed < 0:
                        self.elapsed = 0
                        self.rewind_mode = False
                        self.paused = True
                        self.play_pause_btn.setText("▶ Play")
                        self.rewind_btn.setText("◀◀ Rewind")
                        
                        # Emit signal to animation
                        self.animation_signal.emit({
                            'rewind_mode': False,
                            'paused': True,
                            'position': self.elapsed
                        })
                else:
                    # Normal playback
                    self.elapsed += 0.016 * self.speed  # 60fps * speed
                    if self.elapsed > self.duration:
                        self.elapsed = self.duration
                        self.paused = True
                        self.play_pause_btn.setText("▶ Play")
                        
                        # Emit signal to animation
                        self.animation_signal.emit({
                            'paused': True,
                            'position': self.elapsed
                        })
            
            # Update slider and labels only if user is not currently dragging
            if not self.slider_being_dragged:
                progress_percent = (self.elapsed / self.duration) * 100
                self.progress_slider.setValue(int(progress_percent))
                self.progress_label.setText(f"Progress: {int(progress_percent)}%")
            
            # Update time display
            mins_elapsed = int(self.elapsed) // 60
            secs_elapsed = int(self.elapsed) % 60
            mins_total = int(self.duration) // 60
            secs_total = int(self.duration) % 60
            self.time_label.setText(f"{mins_elapsed}:{secs_elapsed:02d} / {mins_total}:{secs_total:02d}")
    
    def slider_pressed(self):
        self.slider_being_dragged = True
    
    def slider_released(self):
        self.slider_being_dragged = False
        # Apply position change when slider is released
        position_value = self.progress_slider.value()
        self.elapsed = (position_value / 100.0) * self.duration
        # Signal position change to animation
        self.animation_signal.emit({
            'position': self.elapsed
        })
    
    def slider_changed(self, value):
        if self.slider_being_dragged:
            # Preview the position while dragging
            self.elapsed = (value / 100.0) * self.duration
            # Update progress label while dragging
            self.progress_label.setText(f"Progress: {value}%")
    
    def toggle_play_pause(self):
        self.rewind_mode = False
        self.rewind_btn.setText("◀◀ Rewind")
        
        self.paused = not self.paused
        
        if self.paused:
            self.play_pause_btn.setText("▶ Play")
        else:
            self.play_pause_btn.setText("⏸ Pause")
            # If this is the first time hitting play, signal to start animation
            if not self.start_animation:
                self.start_animation = True
        
        # Emit signal to animation
        self.animation_signal.emit({
            'paused': self.paused,
            'rewind_mode': False,
            'position': self.elapsed
        })
    
    def toggle_rewind(self):
        if self.rewind_mode:
            # Stop rewinding
            self.rewind_mode = False
            self.paused = True
            self.rewind_btn.setText("◀◀ Rewind")
            self.play_pause_btn.setText("▶ Play")
        else:
            # Start rewinding
            self.rewind_mode = True
            self.paused = False
            self.rewind_btn.setText("■ Stop")
            self.play_pause_btn.setText("▶ Play")
            # Set animation started flag if first interaction
            if not self.start_animation:
                self.start_animation = True
        
        # Emit signal to animation
        self.animation_signal.emit({
            'rewind_mode': self.rewind_mode,
            'paused': self.paused,
            'position': self.elapsed
        })
    
    def toggle_speed(self):
        if self.speed == 1.0:
            self.speed = 2.0
            self.speed_btn.setText("2x")
        else:
            self.speed = 1.0
            self.speed_btn.setText("1x")
        
        # Emit signal to animation
        self.animation_signal.emit({
            'speed': self.speed
        })
    
    def get_elapsed_normalized(self):
        return min(1.0, max(0.0, self.elapsed / self.duration))

# Initialize Pygame and OpenGL
def initialize_opengl(width, height):
    pygame.init()
    display = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Cube to Sphere Morph Animation with Orbit Control")

    # Set up the OpenGL rendering environment
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (width / height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)
    
    # Enable blending for smooth transition
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Setup background color
    glClearColor(0.0, 0.0, 0.1, 1.0)
    
    return display

# Define cube faces for initial drawing
def cube_vertices(size):
    return [
        [-size, -size, -size],  # 0: bottom left back
        [size, -size, -size],   # 1: bottom right back
        [size, size, -size],    # 2: top right back
        [-size, size, -size],   # 3: top left back
        [-size, -size, size],   # 4: bottom left front
        [size, -size, size],    # 5: bottom right front
        [size, size, size],     # 6: top right front
        [-size, size, size]     # 7: top left front
    ]

def cube_quads():
    return [
        (0, 3, 2, 1),  # Back face
        (4, 5, 6, 7),  # Front face
        (0, 1, 5, 4),  # Bottom face
        (3, 7, 6, 2),  # Top face
        (0, 4, 7, 3),  # Left face
        (1, 2, 6, 5)   # Right face
    ]

# Create a tessellated cube (with subdivision of faces)
def create_tessellated_cube(size=1.0, subdivisions=3):
    vertices = []
    edges = []
    
    # Start with the basic cube
    basic_vertices = cube_vertices(size)
    basic_quads = cube_quads()
    
    # Helper function for interpolation
    def interpolate(v1, v2, t):
        return [v1[0] + (v2[0] - v1[0]) * t,
                v1[1] + (v2[1] - v1[1]) * t,
                v1[2] + (v2[2] - v1[2]) * t]
    
    # Helper function to add vertex if it doesn't exist
    def add_vertex(v):
        for i, existing in enumerate(vertices):
            if np.allclose(v, existing):
                return i
        vertices.append(v)
        return len(vertices) - 1
    
    # Process each face of the cube
    for quad in basic_quads:
        # Get the four corners of this face
        corners = [basic_vertices[i] for i in quad]
        
        # Create a grid of points on this face
        grid_indices = []
        
        for i in range(subdivisions + 1):
            row_indices = []
            t1 = i / subdivisions
            
            for j in range(subdivisions + 1):
                t2 = j / subdivisions
                
                # Interpolate along edges
                edge1 = interpolate(corners[0], corners[1], t2)
                edge2 = interpolate(corners[3], corners[2], t2)
                
                # Interpolate between edges to get point on face
                point = interpolate(edge1, edge2, t1)
                
                # Add vertex
                idx = add_vertex(point)
                row_indices.append(idx)
            
            grid_indices.append(row_indices)
        
        # Create edges from the grid
        for i in range(subdivisions + 1):
            for j in range(subdivisions):
                # Horizontal edge
                edges.append((grid_indices[i][j], grid_indices[i][j+1]))
                
                # Vertical edge (if not last row)
                if i < subdivisions:
                    edges.append((grid_indices[i][j], grid_indices[i+1][j]))
    
    return vertices, edges

# Generate sphere vertices that correspond to cube vertices
def generate_sphere_vertices(cube_vertices, radius=1.0):
    sphere_vertices = []
    for v in cube_vertices:
        # Calculate distance from origin
        distance = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if distance < 0.0001:  # Avoid division by zero
            sphere_vertices.append([0, 0, 0])
        else:
            # Normalize to the sphere radius
            sphere_vertices.append([
                v[0] / distance * radius,
                v[1] / distance * radius,
                v[2] / distance * radius
            ])
    return sphere_vertices

# Linear interpolation between two points
def lerp(start, end, t):
    return [start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t]

# Draw the morphed shape
def draw_morphed_shape(cube_vertices, sphere_vertices, edges, t):
    morph_vertices = []
    
    # Calculate all morphed vertices
    for i in range(len(cube_vertices)):
        morph_vertices.append(lerp(cube_vertices[i], sphere_vertices[i], t))
    
    # Draw the edges of the morphing shape
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(morph_vertices[vertex])
    glEnd()
    
    # Add additional sphere as we get closer to the end
    if t > 0.7:
        opacity = (t - 0.7) / 0.3
        glColor4f(0.5, 0.5, 1.0, opacity * 0.5)
        
        # Draw a smooth sphere with GLU
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_LINE)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, 1.0, 20, 20)
        gluDeleteQuadric(quadric)

# Draw XYZ axis indicator in the top-left corner
def draw_axis_indicator(rotation_x, rotation_y, auto_rotation, elapsed, t):
    # Save the current matrices and viewport
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    
    # Get the current viewport
    viewport = glGetIntegerv(GL_VIEWPORT)
    indicator_size = 80
    
    # Set a new viewport for the indicator (top-left corner)
    glViewport(10, viewport[3] - indicator_size - 10, indicator_size, indicator_size)
    
    # Set up the projection matrix for the indicator
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0, 0.1, 50.0)
    
    # Set up the modelview matrix for the indicator
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -3.0)
    
    # Apply the same rotation as the main scene
    glRotatef(rotation_x, 1, 0, 0)
    glRotatef(rotation_y, 0, 1, 0)
    
    # Apply auto-rotation if enabled
    if auto_rotation:
        rotation_speed = 50 * (1.0 - 0.5 * t)
        rotation_angle = rotation_speed * elapsed
        glRotatef(rotation_angle % 360, 1, 1, 1)
    
    # Draw the axes
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
    
    # Draw small cones at the end of each axis to indicate direction
    # X-axis cone (red)
    glColor3f(1.0, 0.0, 0.0)
    glPushMatrix()
    glTranslatef(1.0, 0.0, 0.0)
    glRotatef(90.0, 0.0, 1.0, 0.0)
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluCylinder(quadric, 0.0, 0.1, 0.2, 8, 1)
    gluDeleteQuadric(quadric)
    glPopMatrix()
    
    # Y-axis cone (green)
    glColor3f(0.0, 1.0, 0.0)
    glPushMatrix()
    glTranslatef(0.0, 1.0, 0.0)
    glRotatef(-90.0, 1.0, 0.0, 0.0)
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluCylinder(quadric, 0.0, 0.1, 0.2, 8, 1)
    gluDeleteQuadric(quadric)
    glPopMatrix()
    
    # Z-axis cone (blue)
    glColor3f(0.0, 0.0, 1.0)
    glPushMatrix()
    glTranslatef(0.0, 0.0, 1.0)
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluCylinder(quadric, 0.0, 0.1, 0.2, 8, 1)
    gluDeleteQuadric(quadric)
    glPopMatrix()
    
    # Restore the original viewport and matrices
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# Function to render Pygame text (called once per frame after OpenGL rendering)
def render_text(display, font, elapsed, t, auto_rotation, momentum_x, momentum_y, width, height):
    # Create a surface for text
    text_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Draw the axis labels
    x_label = font.render("X", True, (255, 50, 50))
    y_label = font.render("Y", True, (50, 255, 50))
    z_label = font.render("Z", True, (50, 50, 255))
    text_surface.blit(x_label, (20, height - 75))
    text_surface.blit(y_label, (35, height - 75))
    text_surface.blit(z_label, (50, height - 75))
    
    # Draw control instructions
    controls = font.render("Left Mouse Button: Orbit | Space: Toggle Auto-Rotation | ESC: Exit", True, (255, 255, 255))
    text_surface.blit(controls, (10, 10))
    
    # Show auto-rotation status
    status = "Auto-Rotation: ON" if auto_rotation else "Auto-Rotation: OFF"
    status_text = font.render(status, True, (255, 255, 255))
    text_surface.blit(status_text, (10, 30))
    
    # Display animation progress
    if elapsed < 30.0:  # duration
        progress = font.render(f"Morphing: {int(t * 100)}%", True, (255, 255, 255))
        text_surface.blit(progress, (10, 50))
    
    # Display momentum for debugging (optional)
    momentum_text = font.render(f"Momentum: X={momentum_x:.2f}, Y={momentum_y:.2f}", True, (200, 200, 200))
    text_surface.blit(momentum_text, (10, 70))
    
    # Blit the text surface onto the display
    display.blit(text_surface, (0, 0))

class FileDialogApp(QMainWindow):
    file_selected_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        from PyQt6.QtWidgets import QFileDialog
        
        # Show file dialog directly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select MP3 file for background music",
            "",
            "MP3 files (*.mp3);;All files (*)"
        )
        
        # Emit the signal with selected file
        self.file_selected_signal.emit(file_name)
        
        # Close the window
        self.close()

class AnimationThread(QThread):
    def __init__(self, audio_file, progress_window):
        super().__init__()
        self.audio_file = audio_file
        self.progress_window = progress_window
        self.running = True
    
    def run(self):
        # Initialize OpenGL and Pygame
        width, height = 800, 600
        display = initialize_opengl(width, height)
        
        # Create a tessellated cube and corresponding sphere
        cube_vertices_data, edges = create_tessellated_cube(size=1.0, subdivisions=5)
        sphere_vertices_data = generate_sphere_vertices(cube_vertices_data)
        
        # Initialize audio player
        audio_player = EnhancedAudioPlayer()
        if self.audio_file:
            audio_player.load_file(self.audio_file)
            print(f"Selected audio file: {self.audio_file}")
        else:
            print("No audio file selected. Animation will play without music.")
        
        # Connect signals from progress window
        self.progress_window.animation_signal.connect(self.handle_progress_signals)
        
        # Setup controller and other variables
        controller = RotationController()
        last_mouse_pos = None
        
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 14)
        
        last_time = time.time()
        clock = pygame.time.Clock()
        animation_completed = False
        
        # Store signals from progress window
        self.paused = True
        self.rewind_mode = False
        self.speed = 1.0
        
        while self.running and self.progress_window.running:
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Update controller
            controller.update(delta_time)
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.progress_window.running = False
                    # Force terminate
                    os._exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.progress_window.running = False
                        # Force terminate
                        os._exit(0)
                    elif event.key == pygame.K_SPACE:
                        if controller.auto_rotation:
                            controller.start_orbit()
                        else:
                            controller.resume_auto_rotation()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        last_mouse_pos = event.pos
                        controller.start_orbit()
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        last_mouse_pos = None
                        controller.stop_orbit()
                
                elif event.type == pygame.MOUSEMOTION:
                    if last_mouse_pos is not None:
                        dx = event.pos[0] - last_mouse_pos[0]
                        dy = event.pos[1] - last_mouse_pos[1]
                        
                        controller.update_mouse_velocity(dx, dy, delta_time)
                        controller.rotation_y.set_target(
                            controller.rotation_y.get() + dx * 0.5)
                        controller.rotation_x.set_target(
                            controller.rotation_x.get() + dy * 0.5)
                        
                        last_mouse_pos = event.pos
            
            # Check if we should resume auto-rotation
            if not controller.auto_rotation and controller.should_resume_auto_rotation():
                controller.resume_auto_rotation()
            
            # Sync audio player with progress window state
            if self.paused:
                if audio_player.playing and not audio_player.paused:
                    audio_player.pause()
            else:
                if self.rewind_mode:
                    if not audio_player.is_rewinding:
                        audio_player.start_rewind()
                else:
                    if audio_player.is_rewinding:
                        audio_player.stop_rewind()
                    if (audio_player.paused or not audio_player.playing) and self.audio_file:
                        audio_player.play(from_pos=self.progress_window.elapsed)
            
            # Sync speed with progress window
            if audio_player.speed != self.speed:
                audio_player.set_speed(self.speed)
            
            # Get animation progress from progress window
            t = self.progress_window.get_elapsed_normalized()
            elapsed = self.progress_window.elapsed
            
            # If animation completed, force auto-rotation
            if t >= 1.0 and not animation_completed:
                animation_completed = True
                if not controller.auto_rotation:
                    controller.resume_auto_rotation()
            
            # Clear the screen and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Reset view
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -5)
            
            # Apply rotations
            glRotatef(controller.rotation_x.get(), 1, 0, 0)
            glRotatef(controller.rotation_y.get(), 0, 1, 0)
            
            # Apply auto-rotation with smooth transition
            if controller.auto_rotation or controller.transition.get() > 0:
                rotation_speed = 50 * (1.0 - 0.5 * t)
                rotation_angle = rotation_speed * elapsed
                auto_rotation_amount = rotation_angle % 360 * controller.transition.get()
                glRotatef(auto_rotation_amount, 1, 1, 1)
            
            # Draw the morphed shape
            draw_morphed_shape(cube_vertices_data, sphere_vertices_data, edges, t)
            
            # Draw the XYZ axis indicator
            draw_axis_indicator(controller.rotation_x.get(), controller.rotation_y.get(), 
                               controller.auto_rotation, elapsed, t)
            
            # Render OpenGL content to the buffer
            glFinish()
            
            # Render Pygame text overlay
            render_text(display, font, elapsed, t, controller.auto_rotation, 
                       controller.momentum_x, controller.momentum_y, width, height)
            
            # Update the display
            pygame.display.flip()
            
            # Control the frame rate - using a higher target to ensure smoother animation
            clock.tick(120)  # Target 120fps to ensure we can maintain 60fps reliably
            
        # Clean up
        audio_player.cleanup()
        pygame.quit()
    
    def handle_progress_signals(self, signal_dict):
        if 'paused' in signal_dict:
            self.paused = signal_dict['paused']
        
        if 'rewind_mode' in signal_dict:
            self.rewind_mode = signal_dict['rewind_mode']
        
        if 'speed' in signal_dict:
            self.speed = signal_dict['speed']

def main():
    # 1) Make your QApplication
    app = QApplication(sys.argv)

    # 2) Immediately ask for the MP3
    file_name, _ = QFileDialog.getOpenFileName(
        None,
        "Select MP3 file for background music",
        "",
        "MP3 files (*.mp3);;All files (*)"
    )

    # 3) Now create your ProgressWindow and AnimationThread
    progress_window = ProgressWindow(duration=30.0)
    animation_thread = AnimationThread(file_name, progress_window)
    animation_thread.start()

    # 4) Finally start the Qt event loop once
    sys.exit(app.exec())

if __name__ == "__main__":
    main()