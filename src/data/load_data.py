import os

# Create a structure
root_dir = "/coco_karpathy"
os.makedirs(f"{root_dir}/images", exist_ok=True)
os.makedirs(f"{root_dir}/annotations", exist_ok=True)

# Download images 2014 (Train + Val) ~13GB + 6GB
# !wget -c http://images.cocodataset.org/zips/train2014.zip -O {root_dir}/train2014.zip
# !unzip -q {root_dir}/train2014.zip -d {root_dir}/images/
# !rm {root_dir}/train2014.zip

# !wget -c http://images.cocodataset.org/zips/val2014.zip -O {root_dir}/val2014.zip
# !unzip -q {root_dir}/val2014.zip -d {root_dir}/images/
# !rm {root_dir}/val2014.zip