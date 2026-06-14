import cv2
import gradio as gr
import numpy as np


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny_edges(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img):
    """Apply a trapezoid mask to focus on the road region."""
    height, width = img.shape[:2]

    polygons = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255 if len(img.shape) == 2 else (255,) * img.shape[2])
    return cv2.bitwise_and(img, mask)


def hough_lines(img):
    """Detect line segments using probabilistic Hough transform."""
    return cv2.HoughLinesP(
        img,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100,
    )


def make_line_points(y1, y2, line_params):
    slope, intercept = line_params
    if abs(slope) < 1e-6:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))


def average_slope_intercept(img, lines):
    """Average detected line segments into one left lane and one right lane."""
    left_fit = []
    right_fit = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    height = img.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    left_lane = make_line_points(y1, y2, np.mean(left_fit, axis=0)) if left_fit else None
    right_lane = make_line_points(y1, y2, np.mean(right_fit, axis=0)) if right_fit else None
    return left_lane, right_lane


def draw_lane_lines(img, left_lane, right_lane, color=(0, 255, 0), thickness=8):
    line_img = np.zeros_like(img)

    if left_lane is not None:
        cv2.line(line_img, left_lane[0], left_lane[1], color, thickness)
    if right_lane is not None:
        cv2.line(line_img, right_lane[0], right_lane[1], color, thickness)

    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0), line_img


def color_threshold(img):
    """Detect white and yellow lane colors in HLS space."""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([15, 30, 115], dtype=np.uint8)
    upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    return cv2.bitwise_or(white_mask, yellow_mask)


def lane_detection_pipeline(img):
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny_edges(blur)
    color_mask = color_threshold(img)
    combined_mask = cv2.bitwise_or(edges, color_mask)
    roi_edges = region_of_interest(combined_mask)
    lines = hough_lines(roi_edges)
    left_lane, right_lane = average_slope_intercept(img, lines)
    final_overlay, _ = draw_lane_lines(img, left_lane, right_lane)

    return {
        "edges": edges,
        "color_mask": color_mask,
        "roi_edges": roi_edges,
        "final_overlay": final_overlay,
        "left_lane": left_lane,
        "right_lane": right_lane,
    }


def to_rgb(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_lanes(input_image):
    if input_image is None:
        raise gr.Error("Please upload a road image first.")

    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    results = lane_detection_pipeline(img_bgr)

    detected_count = int(results["left_lane"] is not None) + int(results["right_lane"] is not None)
    status = f"Detected {detected_count} lane side(s)."
    if detected_count == 0:
        status += " Try using a clearer road image with visible lane markings."

    return (
        to_rgb(results["final_overlay"]),
        to_rgb(results["edges"]),
        to_rgb(results["color_mask"]),
        to_rgb(results["roi_edges"]),
        status,
    )


with gr.Blocks(title="Road Lane Detection") as demo:
    gr.Markdown(
        """
        # Road Lane Detection using Image Analysis

        Upload a road image and this app will detect lane markings using a classical computer vision pipeline:
        grayscale conversion, Gaussian blur, Canny edge detection, color thresholding, region-of-interest masking,
        and Hough line detection.
        """
    )

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload Road Image")
        output_image = gr.Image(type="numpy", label="Detected Lane Overlay")

    run_button = gr.Button("Detect Lanes")
    status = gr.Textbox(label="Detection Status", interactive=False)

    with gr.Accordion("Intermediate Processing Outputs", open=False):
        with gr.Row():
            edges_output = gr.Image(type="numpy", label="Canny Edges")
            color_output = gr.Image(type="numpy", label="Color Mask")
            roi_output = gr.Image(type="numpy", label="ROI Masked Output")

    run_button.click(
        fn=detect_lanes,
        inputs=input_image,
        outputs=[output_image, edges_output, color_output, roi_output, status],
    )

    gr.Markdown(
        """
        ## Note
        This app uses classical image analysis instead of a trained neural network.
        It works best on clear, forward-facing road images with visible lane markings.
        """
    )


if __name__ == "__main__":
    demo.launch()
