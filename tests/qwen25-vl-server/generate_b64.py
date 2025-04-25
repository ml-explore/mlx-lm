import base64
from PIL import Image
import io

# Load your actual test image
img_path = './data/waterfall.png' # Or any valid image path
img = Image.open(img_path)
img = img.convert("RGB") # Ensure RGB

# Save to buffer as JPEG
buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=90)
buffer.seek(0)

# Encode to Base64
b64_bytes = base64.b64encode(buffer.read())
b64_string = b64_bytes.decode('utf-8')

# Create the data URI
data_uri = f"data:image/jpeg;base64,{b64_string}"

print("----- BEGIN BASE64 DATA URI -----")
print(data_uri)
print("----- END BASE64 DATA URI -----")