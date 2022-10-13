import os

parcels = []

for file in os.listdir("Dataset/Ownership/jpg"):
    print(file)
    parcels.append(file[:-4])

parcels.sort()

with open("Dataset/Ownership/buildings.csv", "w") as f:
    for parcel in parcels:
        f.write(f"{parcel},\n")

with open("Dataset/Ownership/total.csv", "w") as f:
    for parcel in parcels:
        f.write(f"{parcel},\n")