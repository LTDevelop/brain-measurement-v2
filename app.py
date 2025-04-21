import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧠 Brain Measurement Tool")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Detection Settings")
    threshold = st.slider("Brightness threshold", 0, 255, 180, help="Higher values = brighter areas only")
    min_area = st.slider("Minimum area (pixels)", 50, 500, 150)
    max_area = st.slider("Maximum area (cm²)", 0.5, 5.0, 2.0, step=0.1)  # New max area control
    blur_size = st.slider("Smoothing", 1, 15, 5, step=2)

def process_image(img, threshold, min_area, max_area, blur_size):
    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
   
    # Brain segmentation (bright areas)
    _, brain_bin = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    brain_bin = cv2.morphologyEx(brain_bin, cv2.MORPH_CLOSE, kernel)
   
    # Detect brains
    contours, _ = cv2.findContours(brain_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ##############################################
    # IMPROVED RULER DETECTION (CRITICAL PART)
    ##############################################
    roi_ruler = gray[-150:, :]  # Bottom 150 pixels
    _, ruler_bin = cv2.threshold(roi_ruler, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   
    # Detect horizontal lines (ruler marks)
    lines = cv2.HoughLinesP(ruler_bin, 1, np.pi/180, threshold=50,
                          minLineLength=50, maxLineGap=10)
   
    scale = None
    if lines is not None:
        # Filter horizontal lines (angle < 5 degrees)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 5:
                horizontal_lines.append(line)
       
        if len(horizontal_lines) >= 2:
            # Calculate distance between first and last mark
            x_coords = [x for line in horizontal_lines for x in [line[0][0], line[0][2]]]
            x_min, x_max = min(x_coords), max(x_coords)
            total_dist_px = x_max - x_min
           
            # Assuming visible ruler shows 10cm
            scale = 10.0 / total_dist_px  # cm/pixel
           
            # Visual debug (optional)
            cv2.rectangle(img, (x_min, img.shape[0]-150),
                         (x_max, img.shape[0]), (255,0,255), 2)
    ##############################################

    # Process results
    result_img = img.copy()
    brains = []
   
    # Sort contours left-to-right before processing
    contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
   
    for i, cnt in enumerate(contours_sorted):
        area_px = cv2.contourArea(cnt)
        area_cm = area_px * (scale ** 2) if scale else 0
       
        # Apply area filters
        if area_px < min_area: continue
        if scale and (area_cm > max_area): continue
       
        x,y,w,h = cv2.boundingRect(cnt)
       
        # Draw with sequential numbering
        cv2.drawContours(result_img, [cnt], -1, (0,255,0), 2)
        cv2.putText(result_img, f"{i+1}", (x,y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
       
        brains.append({
            "ID": i+1,
            "Area (cm²)": round(area_cm, 2),
            "Centroid X": x + w//2,
            "Centroid Y": y + h//2
        })

    return result_img, len(brains), scale, pd.DataFrame(brains)

# Main interface
upload = st.file_uploader("Upload image with 10cm ruler at bottom:",
                         type=["jpg", "png", "tif"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
   
    col1, col2 = st.columns(2)
   
    with col1:
        st.image(img, channels="BGR", caption="Original Image", use_column_width=True)
       
    with col2:
        result = process_image(img, threshold, min_area, max_area, blur_size)
       
        if result:
            processed_img, num_brains, scale, df_results = result
            st.image(processed_img, channels="BGR",
                   caption=f"Detected Brains: {num_brains}",
                   use_column_width=True)
           
            if scale:
                st.success(f"Scale: 1cm = {1/scale:.1f} pixels")
                st.metric("Average Area", f"{df_results['Area (cm²)'].mean():.2f} cm²")
            else:
                st.error("Ruler not detected! Ensure it's visible at the bottom.")
           
            st.dataframe(df_results, hide_index=True)
           
            # Download button
            csv = df_results.to_csv(index=False, sep=";").encode('utf-8')
            st.download_button(
                "📥 Export Results",
                data=csv,
                file_name="brain_areas.csv",
                mime="text/csv"
            )

# Usage instructions
with st.expander("💡 Usage Guide"):
    st.markdown("""
    1. **Position ruler**: Must be at the **bottom** with 1-10cm marks visible
    2. **Lighting**: Dark background with bright brains works best
    3. **Fine-tuning**:
       - Increase **threshold** if detecting too much background
       - Adjust **area limits** to filter fragments/large clusters
    4. **Verify ruler**: Purple area should cover the entire ruler
    """)
