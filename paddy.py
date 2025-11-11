from pathlib import Path
import zipfile
import kaggle
from fastai.vision.all import *
from fastcore.parallel import *
import timm


# Configuration
path = Path('paddy-disease-classification')
dataset_name = 'imbikramsaha/paddy-doctor'
creds = ''
arch = 'convnext_small_in22k'


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


def analyze_image_sizes():
    trn_path = path/'paddy-disease-classification/train_images'
    return trn_path


def create_data_loaders(trn_path):
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75))
    return dls


def print_dataset_info(dls):
    print(f"\nDataset Info:")
    print(f"Classes ({len(dls.vocab)}): {dls.vocab}")
    print(f"Training samples: {len(dls.train_ds)}, Validation samples: {len(dls.valid_ds)}")




def create_model(dls, arch='resnet26d'):
    learn = vision_learner(dls, arch, metrics=error_rate, path='./models').to_fp16()
    return learn


def find_learning_rate(learn):
    lr_find_result = learn.lr_find(suggest_funcs=(valley, slide))
    return lr_find_result


def analyze_learning_rates(lr_find_result):
    conservative_lr = lr_find_result.valley
    aggressive_lr = lr_find_result.slide
    balanced_lr = (conservative_lr + aggressive_lr) / 2
    print(f"Learning Rate: {balanced_lr:.2e} (balanced between {conservative_lr:.2e} and {aggressive_lr:.2e})")
    return balanced_lr


def train_model(dls, epochs):
    learn = create_model(dls)
    lr_find_result = find_learning_rate(learn)
    balanced_lr = analyze_learning_rates(lr_find_result)
    learn.fine_tune(epochs, balanced_lr)
    valid = learn.dls.valid
    preds,targs = learn.get_preds(dl=valid)
    return learn.tta(dl=valid), preds, targs


def print_final_results(learn, tta_preds, preds, targs):
    err = error_rate(tta_preds, targs)
    accuracy = 1 - err.item()
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Error Rate: {err.item():.4f}")

def main():
    download_data()
    set_seed(42)
    
    trn_path = analyze_image_sizes()
    dls = create_data_loaders(trn_path)
    print_dataset_info(dls)
    
    print(f"\nTraining model with {arch} for 3 epochs...")
    learn, tta_preds, preds, targs = train_model(dls, 3)
    print_final_results(learn, tta_preds, preds, targs)

if __name__ == "__main__":
    main()
