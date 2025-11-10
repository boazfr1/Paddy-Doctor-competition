from pathlib import Path
import zipfile
import kaggle
from fastai.vision.all import *
from fastcore.parallel import *
import timm



path = Path('paddy-disease-classification')
dataset_name = 'imbikramsaha/paddy-doctor'
creds = ''


def download_data():
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()
    if not cred_path.exists():
        cred_path.parent.mkdir(exist_ok=True)
        cred_path.write_text(creds)
        cred_path.chmod(0o600)

    if not path.exists():
        print(f"Downloading dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=str(path), unzip=True)
        print("Download completed!")
    else:
        print("Data already exists!")

def get_image_size(image_path):
    return PILImage.create(image_path).size

download_data()
set_seed(42)
trn_path = path/'paddy-disease-classification/train_images'
files = get_image_files(trn_path)

sizes = parallel(get_image_size, files, n_workers=8)
print("sizes", pd.Series(sizes).value_counts())
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(size=224, min_scale=0.75))

# Print dataset info
print(f"Classes: {dls.vocab}")
print(f"Number of classes: {len(dls.vocab)}")
print(f"Training samples: {len(dls.train_ds)}")
print(f"Validation samples: {len(dls.valid_ds)}")

# Get a batch and print its info
batch = dls.one_batch()
x, y = batch
print(f"Batch shape: {x.shape}")
print(f"Labels in batch: {y}")
print(f"Label names: {[dls.vocab[i] for i in y]}")

# Print first few file paths to verify data loading
print(f"Sample file paths:")
for i in range(min(5, len(dls.train_ds))):
    item = dls.train_ds.items[i]
    print(f"  {item}")

learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='./models').to_fp16()

lr_find_result = learn.lr_find(suggest_funcs=(valley, slide))
print(f"\nLearning rate finder results:")
print(f"Valley suggestion: {lr_find_result.valley:.2e}")
print(f"Slide suggestion:  {lr_find_result.slide:.2e}")
print(f"Ratio (slide/valley): {lr_find_result.slide/lr_find_result.valley:.1f}x")

# Calculate and store learning rates for different approaches
conservative_lr = lr_find_result.valley
aggressive_lr = lr_find_result.slide
balanced_lr = (conservative_lr + aggressive_lr) / 2

print(f"\nStored Learning Rates:")
print(f"Conservative LR: {conservative_lr:.2e}")
print(f"Aggressive LR:   {aggressive_lr:.2e}")
print(f"Balanced LR:     {balanced_lr:.2e} (average of both)")

# Analysis
print(f"\nAnalysis:")
print(f"• Conservative LR ({conservative_lr:.2e}) - Use for stable, steady training")
print(f"• Aggressive LR ({aggressive_lr:.2e}) - Use for fast convergence with one-cycle") 
print(f"• Balanced LR ({balanced_lr:.2e}) - Use as a middle-ground approach")
print(f"• The aggressive is {aggressive_lr/conservative_lr:.1f}x higher than conservative")
if aggressive_lr/conservative_lr > 5:
    print(f"• Large ratio suggests the model can handle aggressive learning rates")
else:
    print(f"• Moderate ratio suggests more conservative learning rate scheduling")

learn.fine_tune(3, lr=balanced_lr)

# Check the error rate and loss after training
print(f"\nFinal Training Results:")
val_loss, error_rate = learn.validate()
accuracy = 1 - error_rate

print(f"Validation loss: {val_loss:.4f}")
print(f"Error rate: {error_rate:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
