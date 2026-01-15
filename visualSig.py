import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = "maldroid"  #configure this

rgb_train = np.load(f"./res/{dataset}/RGB_TRAIN.npz")["patches"]
rgb_test = np.load(f"./res/{dataset}/RGB_TEST.npz")["patches"]
att_train = np.load(f"./res/{dataset}/ATT_TRAIN.npz")["patches"]
att_test = np.load(f"./res/{dataset}/ATT_TEST.npz")["patches"]
y_train = np.load(f"./res/{dataset}/RGB_TRAIN.npz")["labels"]
y_test = np.load(f"./res/{dataset}/RGB_TEST.npz")["labels"]

#CONVERT DTYPE TO NUMPY FLOAT32 TO COMPUTE INERTIA VIA NUMPY
att_train = att_train.astype(np.float32)
att_test = att_test.astype(np.float32)          
rgb_train = rgb_train.astype(np.float32)
rgb_test = rgb_test.astype(np.float32)

print("ATT_TRAIN:", att_train.shape, att_train.dtype)
print("ATT_TEST:", att_test.shape, att_test.dtype)
print("RGB_TRAIN:", rgb_train.shape, rgb_train.dtype)
print("RGB_TEST:", rgb_test.shape, rgb_test.dtype)

sig_orig = {}
sig_att = {}

for c in np.unique(y_test):    
    sig_att[c] = att_test[y_test == c].mean(axis=0)
    sig_orig[c] = rgb_test[y_test == c].mean(axis=0)

if dataset == "maldroid":
    classes = ["Benign", "Adware", "Banking", "SMS", "Ransomware"]
elif dataset == "nsl":
    classes = ["Benign", "DoS", "Probe", "R2L", "U2R"]
n = len(classes)

fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

for ax, c, label in zip(axes, sig_att.keys(), classes):
    bar = ax.imshow(sig_att[c].astype(np.uint8))
    ax.set_title(label)
    ax.axis("off")

fig.colorbar(bar, ax=axes, fraction=0.02)
plt.savefig("attention-signatures-test.png", bbox_inches="tight")

def avg_inertia(X, mu):
    diffs = X - mu
    diffs = diffs.reshape(len(X), -1)
    return np.mean(diffs**2)

inertia_orig = {}
inertia_att  = {}

for c in np.unique(y_test):
    inertia_orig[c] = avg_inertia(rgb_test[y_test == c], sig_orig[c])
    inertia_att[c]  = avg_inertia(att_test[y_test == c], sig_att[c])

print(inertia_orig)
print(inertia_att)

avg_distance_maps = {}

for c, mu_c in sig_att.items():
    distance_sum = np.zeros(mu_c.shape[:2])  
    count = 0

    for c2, mu_c2 in sig_att.items():
        if c2 == c:
            continue

        diff = mu_c - mu_c2                  
        dist = np.linalg.norm(diff, axis=-1)  

        distance_sum += dist
        count += 1

    avg_distance_maps[c] = distance_sum / count

all_maps = np.stack(list(avg_distance_maps.values()))  

fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

cmap = cm.plasma.copy()
cmap.set_bad(color='white')

for ax, c, label in zip(axes, sig_att.keys(), classes):
    image = np.ma.masked_where(avg_distance_maps[c] == 0, avg_distance_maps[c])
    bar = ax.imshow(image, vmax=125, cmap=cmap)
    ax.set_title(label)
    ax.axis("off")

fig.colorbar(bar, ax=axes, fraction=0.02)
plt.savefig("inertia.png", bbox_inches="tight")