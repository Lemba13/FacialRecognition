import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT=300
IMAGE_WIDTH=300
BATCH_SIZE=16
PATH = 'fr_siamsese_norm_triplet.pt'
EPOCHS=60

transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #A.augmentations.transforms.Normalize(mean=(0.485), std=(0.229), max_pixel_value=255.0),
    ToTensorV2(),
])
