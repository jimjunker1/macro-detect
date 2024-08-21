from google.cloud import storage
from PIL import image
import io

# Initialize a client
client = storage.Client()

# Get the bucket
bucket_name = 'your-bucket-name'
bucket = client.get_bucket(bucket_name)

# Access a blob (image file) within the bucket
blob = bucket.blob('path/to/your-image.jpg')

# Download the blob's content as bytes
image_bytes = blob.download_as_bytes()

# Convert bytes to an image (e.g., using PIL)
image = Image.open(io.BytesIO(image_bytes))

# Display or process the image
image.show()
