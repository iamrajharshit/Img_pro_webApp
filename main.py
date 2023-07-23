import streamlit as st
import cv2
#from PIL import Image
import numpy as np
#import colorsys
import matplotlib.pyplot as plt

def convert_to_gray(img):
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


#min_threshold max_threshold
def detect_edges(img ,low_threshold, high_threshold):
    edges= cv2.Canny(img, low_threshold, high_threshold)
    return edges

def detect_faces(img):
    face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),6)
    return img


def apply_gaussian_blur(img, k_size):
    # kernel size (e.g., 3x3) (5x5),(7x7),(9x9)
    blurred_image = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return blurred_image



# Function to crop an image with variable parameters
def crop_image(img, x=None, y=None, width=None, height=None):


    # If any of the parameters are not provided, use the full image size
    if x is None:
        x = 0
    if y is None:
        y = 0
    if width is None:
        width = img.shape[1]
    if height is None:
        height = img.shape[0]

    # Ensure the crop region is within the image boundaries
    x = max(0, x)
    y = max(0, y)
    width = min(width, img.shape[1] - x)
    height = min(height, img.shape[0] - y)

    # Crop the image
    cropped_image = img[y:y + height, x:x + width]

    return cropped_image

# Function to perform feature detection and description
def detect_and_describe_features(img, method):
    if img is None:
        st.error("Error: Unable to load the image.")
        return

    # Initialize the feature detector based on the selected method
    if method == 'SIFT':
        detector = cv2.SIFT_create()
    elif method == 'SURF':
        detector = cv2.SURF_create()
    elif method == 'ORB':
        detector = cv2.ORB_create()
    else:
        st.error("Error: Invalid method selected.")
        return

    # Detect and compute the keypoints and descriptors
    keypoints, descriptors = detector.detectAndCompute(img, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

    return image_with_keypoints, keypoints, descriptors


def main():

    # Define the lower and upper bounds for the background color (in HSV)
    #lower_bound = st.sidebar.color_picker("Select the lower bound (background color)")
    #upper_bound = st.sidebar.color_picker("Select the upper bound (background color)")

    #title
    st.title("Image Processing using opencv")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        #convert into open cv image
        file_bytes =np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        img=cv2.imdecode(file_bytes,1)


        #sidebar variables
        k_size = st.sidebar.slider("Select the kernel size for blur:", min_value=3, max_value=9, step=2)
        # Get cropping parameters from the user
        st.sidebar.markdown("Croping parameters")
        x = st.sidebar.number_input("X coordinate:", min_value=0, max_value=img.shape[1] - 1, step=1, value=0)
        y = st.sidebar.number_input("Y coordinate:", min_value=0, max_value=img.shape[0] - 1, step=1, value=0)
        width = st.sidebar.number_input("Width:", min_value=1, max_value=img.shape[1], step=1, value=img.shape[1])
        height = st.sidebar.number_input("Height:", min_value=1, max_value=img.shape[0], step=1, value=img.shape[0])
        # Show hover coordinates in the sidebar
        st.sidebar.markdown(f"Hover Coordinates: ({x + width // 2}, {y + height // 2})")
        # Get the method for feature detection from the user
        method = st.sidebar.selectbox("Select feature detection method:", ['SIFT', 'SURF', 'ORB'])

        # Get the threshold values from the user
        low_threshold = st.sidebar.slider("Low Threshold:", min_value=0, max_value=255, step=1, value=100)
        high_threshold = st.sidebar.slider("High Threshold:", min_value=0, max_value=255, step=1, value=200)
        #original image
        st.image(uploaded_file,use_column_width=True)

        #buttons

        col1,col2,col3=st.columns(3)
        col4, col5,col6=st.columns(3)
        with col1:
            if st.button('Grayscale'):
                con_gray= convert_to_gray(img)
                st.image(con_gray,use_column_width=True)
        with col2:

            if st.button('Detect Edges'):
                con_edges= detect_edges(img,low_threshold, high_threshold)
                st.image(con_edges,use_column_width=True)
        with col3:

            if st.button("Detect Faces"):
                det_face=detect_faces(img)
                st.image(det_face,use_column_width=True)
        with col4:

            if st.button("Blur"):
                st.write(f"Selected kernel size: {k_size}")
                blur_img=apply_gaussian_blur(img,k_size)
                st.image(blur_img,use_column_width=True)
        with col5:

            if st.button("Hover"):
                # Display the original image with hover coordinates
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.scatter(x + width // 2, y + height // 2, c='red', marker='x')
                plt.title(f"Hover Coordinates: ({x + width // 2}, {y + height // 2})")
                plt.axis('off')
                st.pyplot()
        with col6:

            if st.button('Crop Image'):
                crop=crop_image(img, x, y, width, height)
                st.image(crop,use_column_width=True)

        if st.button('detect and describe features'):


            # Perform feature detection and description
            image_with_keypoints, keypoints, descriptors = detect_and_describe_features(img, method)

            # Display the image with keypoints
            st.subheader(f"{method} Keypoints")
            st.image(image_with_keypoints, channels="GRAY", caption=f"{method} Keypoints")

            # Display the number of keypoints detected
            st.write(f"Number of keypoints detected: {len(keypoints)}")

            # Display the first 5 descriptors for illustration purposes
            if descriptors is not None:
                st.subheader(f"{method} Descriptors (First 5)")
                st.write(descriptors[:5])



if __name__ == "__main__":
    main()
