import os
import zipfile

# PyTorch 1.1 moves _download_url_to_file
#   from torch.utils.model_zoo to torch.hub
# PyTorch 1.0 exists another _download_url_to_file
#   2 argument
# TODO: If you remove support PyTorch 1.0 or older,
#       You should remove torch.utils.model_zoo
#       Ref. PyTorch #18758
#         https://github.com/pytorch/pytorch/pull/18758/commits
try:
    from torch.utils.model_zoo import _download_url_to_file
except ImportError:
    from torch.hub import download_url_to_file


def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(path=dest_dir)


if __name__ == '__main__':
    filename='checkpoints.zip'
    download_url_to_file('https://www.dropbox.com/s/y0svnbypzrv4hru/checkpoints.zip?dl=1',filename)
    unzip(filename, '.')