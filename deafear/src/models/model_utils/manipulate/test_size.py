import numpy as np

# Load the NumPy file
data = np.load("/Users/trHien/Python/DeafEar/deafear/src/models/model_utils/manipulate/data_convert/landmarks_D0002.npy")

# Convert to float16
data_float16 = data.astype(np.float16)

# Save the reduced file
np.save("data_float16.npy", data_float16)

print(np.load("data_float16.npy"))
print(f"Original size: {data.nbytes / 1e6:.2f} MB")
print(f"Reduced size: {data_float16.nbytes / 1e6:.2f} MB")
