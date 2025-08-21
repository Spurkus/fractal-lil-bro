import torch
import math
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 20000     # number of points
steps = 100   # iterations

# it's just an equilateral triangle
vertices = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, math.sqrt(3)/2]
], device=device)

# stored in a single tensor for parallel processing and vectorization type schmick
points = torch.rand(N, 2, device=device)

# pytorch processes all N points simultaneously using tensor operations
# parrallel vectorisation type schmick instead of looping individual points
for _ in range(steps):
    # randomly selects {0, 1, 2} for each point (N points)
    v = vertices[torch.randint(0, 3, (N,), device=device)]

    # actually gets the point halfway between the current point and the selected vertex cuz that's how you create st (opposite of ts)
    points = (points + v) / 2

plt.scatter(points[:,0].cpu(), points[:,1].cpu(), s=0.2, color="black")
plt.axis("off")
plt.show()
