import os
import mdshare


def download_ala2_trajs(path=None):
    if not path:
        path = "./ala/"
    if not os.path.exists(path):
        print(f"downloading alanine-dipeptide dataset to {path} ...")

    filenames = [
        os.path.join(f"alanine-dipeptide-{i}-250ns-nowater.xtc") for i in range(3)
    ]

    local_filenames = [
        mdshare.fetch(
            filename,
            working_directory=path,
        )
        for filename in filenames
    ]

    local_filenames.append(mdshare.fetch("alanine-dipeptide-nowater.pdb", working_directory=path))

    return local_filenames

if __name__ == "__main__":
    local_filenames = download_ala2_trajs()
    print(local_filenames)

