import cv2
import numpy as np 
import argparse
from cv2 import aruco
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

from tag_pose_estimation.utils import charuco_board_to_json

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

    save_name_png = output_pdf.replace(".pdf", ".png")

    # Save as PNG for embedding in PDF
    cv2.imwrite(save_name_png, board_img)

    # Create PDF with true-to-size image
    c = canvas.Canvas(output_pdf, pagesize=(img_size_x_mm * mm, img_size_y_mm * mm))
    c.drawImage(save_name_png, 0, 0, width=img_size_x_mm * mm, height=img_size_y_mm * mm)
    c.save()

    # Export the board and dictionary for later use as json
    save_name_json = output_pdf.replace(".pdf", ".json")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a customizable Charuco board and save as PDF.")
    parser.add_argument("--squares_x", type=int, required=True, help="Number of squares in X direction")
    parser.add_argument("--squares_y", type=int, required=True, help="Number of squares in Y direction")
    parser.add_argument("--square_length_m", type=float, required=True, help="Square length in m")
    parser.add_argument("--marker_length_m", type=float, required=True, help="Marker length in m    ")
    parser.add_argument("--aruco_dict", type=str, default="DICT_4X4_50", help="Aruco dictionary name (e.g., DICT_4X4_50)")
    parser.add_argument("--output_pdf", type=str, default="charuco_board.pdf", help="Output PDF file name")
    parser.add_argument("--start_id", type=int, default=0, help="Starting marker ID (default: 0)")
    args = parser.parse_args()

    create_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        aruco_dict_name=args.aruco_dict,
        output_pdf=args.output_pdf,
        start_id=args.start_id
    )