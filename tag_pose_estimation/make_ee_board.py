import cv2
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from PIL import Image

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    num_markers_x = 2
    num_markers_y = 2
    marker_length = 0.04
    marker_separation = 0.003
    start_id = 200

    board = cv2.aruco.GridBoard(
        (
            num_markers_x,  # Number of markers in the X direction.
            num_markers_y,
        ),  # Number of markers in the Y direction.
        marker_length,  # Length of the marker side.
        marker_separation,  # Length of the marker separation.
        dictionary,  # The dictionary of the markers.
        np.array(
            [i + start_id for i in range(num_markers_x * num_markers_y)]
        ),  # (optional) Ids of all the markers (X*Y markers).
    )

    # Compute board size in mm
    board_width_mm = (
        num_markers_x * marker_length + (num_markers_x - 1) * marker_separation
    ) * 1000
    board_height_mm = (
        num_markers_y * marker_length + (num_markers_y - 1) * marker_separation
    ) * 1000

    print(f"Board size: {board_width_mm} mm x {board_height_mm} mm")

    # Create high-res image
    dpi = 300
    pixels_per_mm = dpi / 25.4
    img_width_px = int(board_width_mm * pixels_per_mm)
    img_height_px = int(board_height_mm * pixels_per_mm)

    img = board.generateImage((img_width_px, img_height_px), marginSize=0, borderBits=1)

    # Convert and save as PNG temporarily
    img_pil = Image.fromarray(img)
    tmp_png = "temp_board.png"
    img_pil.save(tmp_png)

    # Create PDF with correct physical size
    pdf_filename = "aruco_board.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=(board_width_mm * mm, board_height_mm * mm))
    c.drawImage(tmp_png, 0, 0, width=board_width_mm * mm, height=board_height_mm * mm)
    c.showPage()
    c.save()

if __name__ == "__main__":
    main()