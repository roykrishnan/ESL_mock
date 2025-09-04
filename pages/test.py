import streamlit as st
import os
from pathlib import Path

st.title("Image Loading Debug Test")
st.markdown("This file will help diagnose why logo.png isn't loading on Streamlit Cloud")

# Test 1: Check current working directory
st.subheader("Test 1: Current Directory Info")
current_dir = os.getcwd()
st.write(f"Current working directory: `{current_dir}`")

# Test 2: List all files in current directory
st.subheader("Test 2: Files in Current Directory")
try:
    files_in_dir = os.listdir('.')
    st.write("Files found:")
    for file in sorted(files_in_dir):
        file_path = Path(file)
        if file_path.is_file():
            file_size = file_path.stat().st_size
            st.write(f"- **{file}** ({file_size} bytes)")
        else:
            st.write(f"- **{file}** (directory)")
    
    # Specifically check for logo.png
    if 'logo.png' in files_in_dir:
        st.success("✅ logo.png found in directory!")
    else:
        st.error("❌ logo.png NOT found in current directory")
        
except Exception as e:
    st.error(f"Error listing directory: {e}")

# Test 3: Check parent directory (in case file is one level up)
st.subheader("Test 3: Files in Parent Directory")
try:
    parent_files = os.listdir('..')
    st.write("Files in parent directory:")
    for file in sorted(parent_files):
        if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            st.write(f"- **{file}** (image file)")
    
    if 'logo.png' in parent_files:
        st.info("📁 logo.png found in parent directory - you'd need `../logo.png`")
        
except Exception as e:
    st.error(f"Error listing parent directory: {e}")

# Test 4: Try different path variations
st.subheader("Test 4: Test Different File Paths")

test_paths = [
    "logo.png",
    "./logo.png", 
    "../logo.png",
    "images/logo.png",
    "../images/logo.png"
]

for path in test_paths:
    try:
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            st.success(f"✅ Found: `{path}` ({file_size} bytes)")
        else:
            st.warning(f"❌ Not found: `{path}`")
    except Exception as e:
        st.error(f"Error checking `{path}`: {e}")

# Test 5: Try loading the image if found
st.subheader("Test 5: Image Loading Tests")

for path in test_paths:
    if os.path.exists(path):
        st.write(f"**Testing image load for: `{path}`**")
        
        # Test 5a: Standard st.image
        try:
            st.image(path, caption=f"Loaded with st.image('{path}')", width=200)
            st.success(f"✅ st.image() worked for `{path}`")
        except Exception as e:
            st.error(f"❌ st.image() failed for `{path}`: {type(e).__name__} - {str(e)}")
        
        # Test 5b: st.sidebar.image
        try:
            st.sidebar.image(path, caption=f"Sidebar test: {path}", width=200)
            st.success(f"✅ st.sidebar.image() worked for `{path}`")
        except Exception as e:
            st.error(f"❌ st.sidebar.image() failed for `{path}`: {type(e).__name__} - {str(e)}")
        
        # Test 5c: Read file manually
        try:
            with open(path, 'rb') as f:
                image_data = f.read()
            st.image(image_data, caption=f"Manual read: {path}", width=200)
            st.success(f"✅ Manual file reading worked for `{path}`")
        except Exception as e:
            st.error(f"❌ Manual file reading failed for `{path}`: {type(e).__name__} - {str(e)}")
        
        break  # Only test the first found image

# Test 6: File system permissions
st.subheader("Test 6: File Permissions Check")
if os.path.exists('logo.png'):
    try:
        # Check if file is readable
        with open('logo.png', 'rb') as f:
            first_bytes = f.read(10)
        st.success(f"✅ File is readable. First 10 bytes: {first_bytes}")
        
        # Check file stats
        stats = os.stat('logo.png')
        st.write(f"File size: {stats.st_size} bytes")
        st.write(f"File mode: {oct(stats.st_mode)}")
        
    except Exception as e:
        st.error(f"❌ File permission/access error: {e}")

# Test 7: GitHub repository structure check
st.subheader("Test 7: Expected vs Actual Structure")
st.write("**Expected file structure for your app:**")
st.code("""
your-repo/
├── pages/
│   └── 3_Dynamic Report - Owner.py  (your main file)
│   └── logo.png  (if logo should be with the Python file)
└── logo.png  (if logo should be at repo root)
""")

st.markdown("**Next steps based on results:**")
st.markdown("""
- If logo.png shows ✅ in Test 2 but fails in Test 5: **File format issue**
- If logo.png shows ❌ in Test 2 but ✅ in Test 3: **Use `../logo.png` in your code**
- If logo.png shows ❌ in both: **File not committed to GitHub properly**
- If everything shows ✅ but still fails in your main app: **Streamlit caching issue**
""")

st.info("💡 **Tip:** Take a screenshot of this page and share the results to get targeted help!")
