import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QSurfaceFormat

from modules.ui import MainWindow

def setup_opengl_format():
    """Set up the OpenGL format for the application"""
    gl_format = QSurfaceFormat()
    gl_format.setVersion(2, 1)  # Using OpenGL 2.1 for compatibility
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    gl_format.setSamples(4)  # 4x MSAA
    QSurfaceFormat.setDefaultFormat(gl_format)

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Set up OpenGL format
    setup_opengl_format()
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()