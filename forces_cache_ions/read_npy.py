import numpy as np

# Load the file
data = np.load("./ionic_forces.npy", allow_pickle=True).item()

# # Print the first 10 items
# for i, (key, value) in enumerate(data.items()):
#     print(f"{key}: {value}")
#     if i >= 9:
#         break

print(data[5331].get(1403))
print(data.get((5331, 1403)))
