
import os
import streamlit as st
import base64
from PIL import Image

# Define the directory path
directory_path = r'Home_GUI\static\img\plates'

# Use the os module to list all files in the directory
files = os.listdir(directory_path)

# Filter out directories and other non-file items
image_files = [fr"Home_GUI\static\img\plates\{f}" for f in files if os.path.isfile(os.path.join(directory_path, f))]


# Generate the HTML code for the images
image_html = ""

    
for file in image_files:
    if file.endswith(".jpg") or file.endswith(".png"):
        
        # image_path = os.path.join(IMAGE_FOLDER, file)
        
        file_ = open(file, 'rb')
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        image_html += f'<img src="data:image/gif;base64,{data_url}" alt="{file}">'
        
# Generate the HTML page with the images and scrollbar
html = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        #scrollable {{
            width: 150px;
            height: 300px;
            top: 10px;
            left: 450px;
            position: absolute;
            background-color: #000000;
            overflow-y: scroll;
        }}
        img {{
            width: 100px;
            max-width: 300px;
            height: 300px;
            margin: 10px;
        }}
    </style>
</head>
<body>
    <div id="scrollable">
        {image_html}
    </div>
</body>
</html>
'''

# Display the HTML page in Streamlit
st.write(html, unsafe_allow_html=True)