from pathlib import Path
import datetime

import cv2
import numpy as np 
from cv2 import aruco
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

from tag_pose_estimation.utils import (
    charuco_board_to_json,
    get_project_root,
)

def create_charuco_board(
        squares_x: int, 
        squares_y: int, 
        square_length_m: float, 
        marker_length_m: float, 
        aruco_dict_name: str, 
        output_pdf: str,
        start_id: int = 0
    ) -> None:
    # Get the aruco dictionary
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict_name))
    if start_id != 0:
        # Compute the number of white chessboard squares for a chess board pattern
        # of size (squares_x, squares_y) with the first square being black
        black_squares = (squares_x // 2) * (squares_y // 2) + \
                            ((squares_x + 1) // 2) * ((squares_y + 1) // 2)
        white_squares = (squares_x * squares_y) - black_squares
        num_aruco_markers = white_squares
        id_arr = np.array([i + start_id for i in range(num_aruco_markers)])
    else:
        id_arr = None
    # Create the Charuco board
    board = aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length_m,
        markerLength=marker_length_m,
        dictionary=aruco_dict,
        ids=id_arr
    )
    # Draw the board as an image
    # Create high-res image
    dpi = 300
    pixels_per_mm = dpi / 25.4

    img_size_x_mm = squares_x * square_length_m * 1000  # convert meters to millimeters
    img_size_y_mm = squares_y * square_length_m * 1000  # convert meters to millimeters
    img_size_x_px = int(img_size_x_mm * pixels_per_mm)
    img_size_y_px = int(img_size_y_mm * pixels_per_mm)
    board_img = board.generateImage((img_size_x_px, img_size_y_px))

    # Compute board size in mm
    print(f"Board size: {img_size_x_mm} mm x {img_size_y_mm} mm")

    # Output file path
    calibration_folder = get_calibration_board_save_folder()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_folder = calibration_folder / f"{timestamp}_charuco_board"
    save_folder.mkdir()

    # Save as PNG for embedding in PDF
    save_name_png = save_folder / "charuco_board.png"
    cv2.imwrite(str(save_name_png), board_img)

    # Create PDF with true-to-size image
    save_name_pdf = save_folder /"charuco_board.pdf"
    c = canvas.Canvas(str(save_name_pdf), pagesize=(img_size_x_mm * mm, img_size_y_mm * mm))
    c.drawImage(save_name_png, 0, 0, width=img_size_x_mm * mm, height=img_size_y_mm * mm)
    c.save()

    # Export the board and dictionary for later use as json
    save_name_json = save_folder / "charuco_board.json"
    charuco_board_to_json(
        board,
        squares_x,
        squares_y,
        square_length_m,
        marker_length_m,
        aruco_dict_name,
        save_file=save_name_json,
    )

    return None

def get_calibration_board_save_folder() -> Path:
    project_root = get_project_root()
    calibration_folder = project_root / "config" / "calibration_boards"

    # Create the directory if it doesn't exist
    if not calibration_folder.parent.exists():
        calibration_folder.parent.mkdir()

    if not calibration_folder.exists():
        calibration_folder.mkdir()
    
    return calibration_folder