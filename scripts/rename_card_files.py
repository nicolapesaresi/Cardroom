import os
import glob
import tqdm

TARGET_DIR = "/Users/nicola/Desktop/vscode/Cardroom/cardroom/assets/cards"
CARDS = "french"

bergamasche_suit = {
    "1": "bastoni",
    "2": "spade",
    "3": "coppe",
    "4": "ori",
}

def rename_bergamasche():
    for file in tqdm.tqdm(glob.glob(os.path.join(TARGET_DIR, "bergamasche","*.jpg"))):
        oldname = os.path.splitext(os.path.basename(file))[0]
        face = oldname.split("-")[3]
        suit = bergamasche_suit[oldname.split("-")[1]]
        newname = f"{suit}_{face}.png"
        newpath = os.path.abspath(os.path.dirname(file))
        os.rename(file, os.path.join(newpath, newname))

def rename_napoletane():
    for file in tqdm.tqdm(glob.glob(os.path.join(TARGET_DIR, "napoletane", "*.png"))):
        oldname = os.path.splitext(os.path.basename(file))[0] 
        if oldname.endswith("10"):
            suit = oldname[:-2] 
            face = "10"
        else:
            suit = oldname[:-1]  
            face = oldname[-1]

        newname = f"{suit}_{face}.jpg"
        newpath = os.path.dirname(file)
        os.rename(file, os.path.join(newpath, newname))

def rename_french():
    for file in tqdm.tqdm(glob.glob(os.path.join(TARGET_DIR, "french","*.png"))):
        oldname = os.path.splitext(os.path.basename(file))[0]
        if oldname.endswith("2"):
            oldname = oldname[:-1]
        face = oldname.split("_")[0]
        suit = oldname.split("_")[2]
        newname = f"{suit}_{face}.png"
        newpath = os.path.abspath(os.path.dirname(file))
        os.rename(file, os.path.join(newpath, newname))

if __name__ == "__main__":
    if CARDS == "bergamasche":
        rename_bergamasche()
        print("Done.")
    if CARDS == "napoletane":
         rename_napoletane()
         print("Done.")
    if CARDS == "french":
        rename_french()
        print("Done.")