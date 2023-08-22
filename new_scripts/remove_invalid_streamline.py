import nibabel as nib

import argparse
import os

from dipy.io.streamline import load_tractogram, save_tractogram

def build_args_parser():
    p = argparse.ArgumentParser(description="Hi",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram', metavar='TRACTS',
                   help='Tractogram file. File must be tck or trk.')

    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    tractogram = args.tractogram

    if not os.path.isfile(tractogram):
        parser.error('"{0}" must be a file!'.format(tractogram))

    _, ext = os.path.splitext(tractogram)
    if not (ext == '.tck' or ext == '.trk'):
        parser.error("Tractogram file should be a .tck or .trk, not {}"
                     .format(ext))

    DATASET_FOLDER='/home/awuxingh/data/fibercup'

    ref_anat_filename = DATASET_FOLDER + "/fibercup/dwi/fibercup_b0_dwi.nii.gz"
    ref_anatomy = nib.load(ref_anat_filename)

    trk_filename = tractogram
    tract = load_tractogram(trk_filename, ref_anatomy, bbox_valid_check=False)
    print(tract.is_bbox_in_vox_valid())

    tract.remove_invalid_streamlines()
    print(tract.is_bbox_in_vox_valid())
    save_tractogram(tract, os.path.dirname(tractogram) + "/tractogram_fibercup_invalid_removed.trk")

if __name__ == "__main__":
    main()
