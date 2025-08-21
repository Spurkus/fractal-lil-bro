import torch
import matplotlib.pyplot as plt

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resolution and domain
res = 2000

x = torch.linspace(0, 1, res, device=device)
y = torch.linspace(0, 1, res, device=device)

Y, X = torch.meshgrid(y, x, indexing="ij")

# start with all white
carpet = torch.ones_like(X, dtype=torch.float32, device=device)

# number of iterations (depth of recursion)
depth = 6

# hausdorff dimension is log(8) / log(3) ~ 1.8928 because each iteration removes 1/3 of the area and
# replaces it with 8 smaller squares, each of which is 1/3 the size of the previous ones.
# loop over scales
for i in range(depth):
    scale = 3**i

    # compute which cells are in the middle square
    xi = ((X * scale).long() % 3 == 1)
    yi = ((Y * scale).long() % 3 == 1)

    mask = xi & yi
    carpet[mask] = 0.0  # remove middle square

carpet_np = carpet.cpu().numpy()
plt.figure(figsize=(8, 8))
plt.imshow(carpet_np, cmap="binary", origin="lower")
plt.axis("off")
plt.tight_layout()
plt.show()
