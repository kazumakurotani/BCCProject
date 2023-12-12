import setup_window


def open_window_image_analysis(HomeWindow):
    """
    open image analysis window

    Args:
        HomeWindow (class): manage home window
    """
    setup_window.imageAnalysisWindow = setup_window.ImageAnalysisWindow()
    setup_window.imageAnalysisWindow.show()
