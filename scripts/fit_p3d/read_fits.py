from lace_pk.input_emu import get_input_emulator


def main():
    folder_input = "/data/desi/scratch/jchavesm/p3d_fits/"
    ntot = 1023
    file_out = "/data/desi/scratch/jchavesm/LaCE_pk/lace_pk/data/input_emu"
    get_input_emulator(folder_input, ntot, file_out)


if __name__ == "__main__":
    main()
