import hashlib
import os
import zipfile

from srcs.utils import error_exit

ZIP_PATH = "leaffliction.zip"
SIGNATURE_PATH = "signature.txt"


def make_zip(model_path, data_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, os.path.basename(model_path))
        for root, _, files in os.walk(data_dir):
            for name in files:
                full = os.path.join(root, name)
                archive.write(full, full)
    print(f"Zip created : {zip_path}")


def write_signature(zip_path, signature_path):
    sha1 = hashlib.sha1()
    with open(zip_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha1.update(chunk)
    digest = sha1.hexdigest()
    with open(signature_path, "w") as handle:
        handle.write(digest + "\n")
    print(f"signature sha1 : {digest}")
    return digest


def run(model_path, data_dir):
    if not os.path.isfile(model_path):
        error_exit(f"model '{model_path}' missing")
    if not os.path.isdir(data_dir):
        error_exit(f"'{data_dir}' is not a directory")
    make_zip(model_path, data_dir, ZIP_PATH)
    write_signature(ZIP_PATH, SIGNATURE_PATH)
