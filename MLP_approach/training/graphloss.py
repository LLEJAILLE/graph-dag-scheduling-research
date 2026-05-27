import re
import matplotlib.pyplot as plt

LOG_FILE = "epochs_loss.txt"

epochs = []
losses = []

pattern = re.compile(r"Epoch\s+(\d+),\s+Loss:\s+([0-9.eE+-]+)")

with open(LOG_FILE, "r", encoding="utf-16") as f:
    for line in f:
        match = pattern.search(line)

        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))

            epochs.append(epoch)
            losses.append(loss)

print(f"Loaded {len(losses)} epochs.")

plt.figure(figsize=(14, 7))

plt.plot(
    epochs,
    losses,
    linewidth=1.5,
    label="Training Loss"
)

plt.title("Cross-Entropy Loss over Training Epochs", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)

plt.legend()


plt.tight_layout()

plt.savefig("training_loss.png", dpi=300)

plt.show()